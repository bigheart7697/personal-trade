package strategies

import (
	"fmt"
	"math"
	"sort"

	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// trials: counted mechanically per walk-forward run — grid is m x k (2x2 =
// 4 combos/fold); libSize (the candidate-library cap) is fixed at 1000, the
// strategy's definition rather than a fit parameter, the same convention as
// ml-logit's refitEvery and vol-target's checkEvery.

// mlKNN is a k-nearest-neighbor pattern matcher: it shape-matches the most
// recent m daily returns against a library of historical m-day return
// shapes and goes long when those shapes' actual next-day outcomes were
// reliably positive. Unlike ml-logit/ml-boost it fits no parametric model at
// all — "training" IS the candidate library, rebuilt fresh from history on
// every call — so there is no cached-model cadence to manage; recomputing
// it every bar is O(libSize * m), cheap even at daily-bar scale over a
// multi-thousand-bar backtest (see TargetWeights).
//
// NO LOOKAHEAD: a candidate pattern at bar j pairs an m-return shape built
// from closes[j-m+1..j] with the label close[j+1]/close[j]-1 (bar j's own
// forward return); only candidates with j <= t-1 (t = the current bar, the
// last element of ctx.History()) ever enter the library, so a candidate's
// label always refers to a bar no later than t. The query pattern is built
// from closes[t-m+1..t] — bar t's own trailing shape, using nothing beyond
// bar t.
//
// DETERMINISM: no randomness. Ties in nearest-neighbor distance are broken
// by preferring the MORE RECENT candidate (higher j), a fixed rule applied
// via a fully-specified sort comparator (see TargetWeights).
type mlKNN struct {
	m       int
	k       int
	libSize int
}

func newMLKNN() *mlKNN {
	return &mlKNN{m: 5, k: 10, libSize: 1000}
}

func (s *mlKNN) Name() string { return "ml-knn" }

func (s *mlKNN) Description() string {
	return "k-NN pattern matcher: finds the k most similar m-day return shapes in recent history and goes long when their next-day outcomes were reliably positive."
}

func (s *mlKNN) Horizon() strategy.Horizon { return strategy.Short }

// WarmupBars needs at least k candidate patterns available in the library
// by the time trading starts, plus m bars to compute the query pattern
// itself. With t = warmup on the engine's first OnBar call (see
// donchian.go's WarmupBars comment for the exact convention), candidates
// range over j in [m, t-1] — t-m of them — so WarmupBars = m+k makes
// exactly k candidates available on the very first call.
func (s *mlKNN) WarmupBars() int { return s.m + s.k }

// mlKNNPattern returns the z-scored m-return shape ending at bar i: the m
// daily returns closes[i-m+1..i]/closes[i-m..i-1]-1, standardized by THIS
// WINDOW'S OWN mean/std (shape matching, not any global or cross-window
// statistic). ok is false when there isn't enough history (i < m), a zero
// close would divide by zero, or the window is degenerate (zero std — a
// perfectly flat segment carries no discriminating shape, so it is skipped
// rather than dividing by zero).
func mlKNNPattern(closes []float64, i, m int) ([]float64, bool) {
	if i < m || i >= len(closes) {
		return nil, false
	}
	rets := make([]float64, m)
	for k := 0; k < m; k++ {
		day := i - m + 1 + k
		if closes[day-1] == 0 {
			return nil, false
		}
		rets[k] = closes[day]/closes[day-1] - 1
	}

	var sum float64
	for _, r := range rets {
		sum += r
	}
	mean := sum / float64(m)
	var sq float64
	for _, r := range rets {
		d := r - mean
		sq += d * d
	}
	std := math.Sqrt(sq / float64(m))
	if std == 0 {
		return nil, false
	}

	out := make([]float64, m)
	for idx, r := range rets {
		out[idx] = (r - mean) / std
	}
	return out, true
}

// euclidDist returns the Euclidean distance between two equal-length
// vectors.
func euclidDist(a, b []float64) float64 {
	var sum float64
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}

// mlKNNNeighbor is one candidate pattern in the library: its bar index j
// (needed for the recency tie-break), its label (j's own forward return),
// and its distance to the current query pattern.
type mlKNNNeighbor struct {
	j     int
	label float64
	dist  float64
}

// TargetWeights is the pure-history brain: build the query pattern at the
// current bar, find its k nearest neighbors among the most recent libSize
// candidate patterns, and go long only when the mean neighbor forward
// return is positive AND at least ceil(60%) of the neighbors individually
// had a positive forward return (the majority filter, which discounts a
// prediction driven by one or two large outlier neighbors). It never reads
// ctx.Portfolio().
func (s *mlKNN) TargetWeights(ctx *strategy.Context) map[string]float64 {
	history := ctx.History()
	if len(history) == 0 {
		return map[string]float64{}
	}
	sym := history[len(history)-1].Symbol
	closes := strategy.Closes(history)
	t := len(closes) - 1

	query, ok := mlKNNPattern(closes, t, s.m)
	if !ok {
		return map[string]float64{}
	}

	hi := t - 1 // latest candidate index: label needs closes[hi+1] = closes[t]
	if hi < s.m {
		return map[string]float64{}
	}
	lo := hi - s.libSize + 1 // restrict the library to the most recent libSize candidates
	if lo < s.m {
		lo = s.m
	}

	candidates := make([]mlKNNNeighbor, 0, hi-lo+1)
	for j := lo; j <= hi; j++ {
		pat, patOK := mlKNNPattern(closes, j, s.m)
		if !patOK {
			continue
		}
		label := closes[j+1]/closes[j] - 1
		candidates = append(candidates, mlKNNNeighbor{j: j, label: label, dist: euclidDist(query, pat)})
	}
	if len(candidates) < s.k {
		return map[string]float64{}
	}

	sort.Slice(candidates, func(a, b int) bool {
		if candidates[a].dist != candidates[b].dist {
			return candidates[a].dist < candidates[b].dist
		}
		return candidates[a].j > candidates[b].j // tie-break: prefer the more recent candidate
	})

	neighbors := candidates[:s.k]
	var sum float64
	posCount := 0
	for _, nb := range neighbors {
		sum += nb.label
		if nb.label > 0 {
			posCount++
		}
	}
	prediction := sum / float64(s.k)
	need := int(math.Ceil(0.6 * float64(s.k)))

	if prediction > 0 && posCount >= need {
		return map[string]float64{sym: 1.0}
	}
	return map[string]float64{}
}

// OnBar derives the edge-triggered signal from TargetWeights (the
// pure-history brain) and the ACTUAL portfolio position — never a
// remembered "last signal sent" (the same derive-don't-shadow discipline as
// every other strategy in this package).
func (s *mlKNN) OnBar(ctx *strategy.Context, bar domain.Bar) []domain.Signal {
	target := s.TargetWeights(ctx)[bar.Symbol]

	qty := int64(0)
	if pos, ok := ctx.Portfolio().Positions[bar.Symbol]; ok {
		qty = pos.Qty
	}
	flat := qty == 0

	if target > 0 && flat {
		return []domain.Signal{{Symbol: bar.Symbol, TargetWeight: target}}
	}
	if target == 0 && !flat {
		return []domain.Signal{{Symbol: bar.Symbol, TargetWeight: 0.0}}
	}
	return nil
}

// ParamSpace declares the discrete grid walk-forward optimization searches
// over: 2 pattern lengths x 2 neighbor counts (4 combos; no cross-parameter
// constraint excludes any of them).
func (s *mlKNN) ParamSpace() []strategy.ParamDef {
	return []strategy.ParamDef{
		{Name: "m", Values: []float64{5, 10}},
		{Name: "k", Values: []float64{10, 25}},
	}
}

// WithParams returns a fresh *mlKNN starting from the 5/10 defaults,
// overridden by any m/k entries in params. It never mutates the receiver.
// Unknown keys, non-integral values, m < 2, and k < 1 are all rejected.
//
// libSize (the candidate-library cap) is deliberately NOT tunable — the
// same convention as ml-logit's refitEvery and vol-target's checkEvery: it
// is part of this strategy's definition (how much history the library
// looks back over), not a curve-fitted parameter.
func (s *mlKNN) WithParams(params map[string]float64) (strategy.Strategy, error) {
	next := newMLKNN()

	for name, v := range params {
		switch name {
		case "m":
			if v != math.Trunc(v) {
				return nil, fmt.Errorf("ml-knn: m must be an integral value, got %v", v)
			}
			next.m = int(v)
		case "k":
			if v != math.Trunc(v) {
				return nil, fmt.Errorf("ml-knn: k must be an integral value, got %v", v)
			}
			next.k = int(v)
		default:
			return nil, fmt.Errorf("ml-knn: unknown parameter %q", name)
		}
	}

	if next.m < 2 {
		return nil, fmt.Errorf("ml-knn: m must be >= 2, got %d", next.m)
	}
	if next.k < 1 {
		return nil, fmt.Errorf("ml-knn: k must be >= 1, got %d", next.k)
	}

	return next, nil
}

var _ strategy.Tunable = (*mlKNN)(nil)
var _ strategy.TargetWeighter = (*mlKNN)(nil)

func init() {
	strategy.Register(func() strategy.Strategy { return newMLKNN() })
}
