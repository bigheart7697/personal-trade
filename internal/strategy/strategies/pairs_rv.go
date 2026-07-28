package strategies

import (
	"fmt"
	"math"
	"time"

	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// trials: counted mechanically per walk-forward run — grid is lookback x
// entryZ x exitZ (2x2x2 = 8 combos/fold); WithParams's cross-param
// constraint (0 < exitZ < entryZ) accepts all 8 combos here (no combo
// excluded, since every entryZ candidate exceeds every exitZ candidate).

// pairsRV is a relative-value pairs strategy over a spread of two symbols'
// log prices.
//
// IMPORTANT deviation from classic pairs trading: this platform is
// LONG-ONLY (risk.Manager zeroes any short request — cardinal rule 2's
// manager owns that gate, not this strategy), so a classic hedged
// long/short pair position is not available here. This is instead a
// relative-value ROTATION: it holds whichever leg is currently RELATIVELY
// CHEAP outright (target weight 1.0), or holds nothing, rather than being
// long one leg and short the other. It captures the same "spread reverts to
// its mean" thesis directionally, one leg at a time, at the cost of losing
// the hedge's market-neutrality.
//
// Derive-don't-shadow: the strategy holds no mutable state. Spread and
// z-score are recomputed from history every call, and "which leg (if
// either) is currently held" is derived from the portfolio's actual
// positions, never remembered — so a rejected or unfilled order can never
// desynchronize the strategy from reality.
type pairsRV struct {
	universe []string // exactly two symbols: {A, B}
	lookback int
	entryZ   float64
	exitZ    float64
}

// defaultPairsUniverse is the default two-symbol universe: SPY and QQQ, both
// broad US-equity ETFs with a historically tight, mean-reverting log-price
// spread.
var defaultPairsUniverse = []string{"SPY", "QQQ"}

func newPairsRV() *pairsRV {
	return &pairsRV{
		universe: append([]string(nil), defaultPairsUniverse...),
		lookback: 60,
		entryZ:   2.0,
		exitZ:    0.5,
	}
}

func (s *pairsRV) Name() string { return "pairs-rv" }

func (s *pairsRV) Description() string {
	return "Relative-value rotation: hold whichever of two symbols is cheap on a mean-reverting log-price spread z-score, or nothing (long-only, no hedge)."
}

func (s *pairsRV) Horizon() strategy.Horizon { return strategy.Short }

// WarmupBars needs lookback+1 bars of the spread series (lookback samples
// for the mean/stdev, plus the current bar) on each leg.
func (s *pairsRV) WarmupBars() int { return s.lookback + 1 }

func (s *pairsRV) Universe() []string { return s.universe }

func (s *pairsRV) symA() string { return s.universe[0] }
func (s *pairsRV) symB() string { return s.universe[1] }

// OnBar is the single-symbol Strategy interface stub. The engine never
// calls this for a MultiSymbol strategy — see the documented convention on
// strategy.MultiSymbol — but pairsRV must still satisfy Strategy.
func (s *pairsRV) OnBar(ctx *strategy.Context, bar domain.Bar) []domain.Signal {
	return nil
}

// spreadSeries returns ln(closeA) - ln(closeB) for the last lookback+1
// aligned bars of both legs, oldest first, and ok=false if either leg has
// insufficient history or the histories can't be aligned by length.
func (s *pairsRV) spreadSeries(histA, histB []domain.Bar) ([]float64, bool) {
	if len(histA) < s.lookback+1 || len(histB) < s.lookback+1 {
		return nil, false
	}

	n := s.lookback + 1
	winA := histA[len(histA)-n:]
	winB := histB[len(histB)-n:]

	spread := make([]float64, n)
	for i := 0; i < n; i++ {
		if winA[i].Close <= 0 || winB[i].Close <= 0 {
			return nil, false
		}
		spread[i] = math.Log(winA[i].Close) - math.Log(winB[i].Close)
	}
	return spread, true
}

func (s *pairsRV) OnUniverseBar(ctx *strategy.Context, date time.Time, bars map[string]domain.Bar) []domain.Signal {
	histA := ctx.HistoryOf(s.symA())
	histB := ctx.HistoryOf(s.symB())

	spread, ok := s.spreadSeries(histA, histB)
	if !ok {
		return nil
	}

	// Mean/stdev over the lookback samples PRIOR to the current one, then
	// compare the current spread against that baseline — same
	// exclude-current-bar discipline as donchian's channel, applied to the
	// spread series rather than price.
	prior := spread[:len(spread)-1]
	now := spread[len(spread)-1]

	mean := 0.0
	for _, v := range prior {
		mean += v
	}
	mean /= float64(len(prior))

	sd, ok := strategy.StdDev(prior, len(prior))
	if !ok || sd == 0 {
		return nil
	}

	z := (now - mean) / sd

	// Derive held state from the actual portfolio, never remembered.
	heldA := false
	if pos, ok := ctx.Portfolio().Positions[s.symA()]; ok && pos.Qty != 0 {
		heldA = true
	}
	heldB := false
	if pos, ok := ctx.Portfolio().Positions[s.symB()]; ok && pos.Qty != 0 {
		heldB = true
	}

	var signals []domain.Signal

	switch {
	case z < -s.entryZ:
		// A is cheap relative to B: hold A outright, exit B.
		if !heldA {
			signals = append(signals, domain.Signal{Symbol: s.symA(), TargetWeight: 1.0})
		}
		if heldB {
			signals = append(signals, domain.Signal{Symbol: s.symB(), TargetWeight: 0.0})
		}
	case z > s.entryZ:
		// B is cheap relative to A: hold B outright, exit A.
		if !heldB {
			signals = append(signals, domain.Signal{Symbol: s.symB(), TargetWeight: 1.0})
		}
		if heldA {
			signals = append(signals, domain.Signal{Symbol: s.symA(), TargetWeight: 0.0})
		}
	case math.Abs(z) < s.exitZ:
		// Spread has reverted: flatten both legs. Only emit exits for
		// whichever leg is actually held — no churn on an already-flat leg.
		if heldA {
			signals = append(signals, domain.Signal{Symbol: s.symA(), TargetWeight: 0.0})
		}
		if heldB {
			signals = append(signals, domain.Signal{Symbol: s.symB(), TargetWeight: 0.0})
		}
	default:
		// Between exitZ and entryZ: hysteresis band, hold whatever is held.
		// No signals.
	}

	return signals
}

// ParamSpace declares the discrete grid walk-forward optimization searches
// over: 2 lookback windows x 2 entry-z thresholds x 2 exit-z thresholds (8
// combos; every entryZ candidate exceeds every exitZ candidate so none of
// the 8 combos are excluded by WithParams's cross-param constraint).
func (s *pairsRV) ParamSpace() []strategy.ParamDef {
	return []strategy.ParamDef{
		{Name: "lookback", Values: []float64{40, 60}},
		{Name: "entryZ", Values: []float64{1.5, 2.0}},
		{Name: "exitZ", Values: []float64{0.25, 0.5}},
	}
}

// WithParams returns a fresh *pairsRV starting from the defaults, overridden
// by any lookback/entryZ/exitZ entries in params. It never mutates the
// receiver. Unknown keys, non-integral or <10 lookback, and any combination
// that doesn't satisfy 0 < exitZ < entryZ are all rejected.
func (s *pairsRV) WithParams(params map[string]float64) (strategy.Strategy, error) {
	next := newPairsRV()

	for name, v := range params {
		switch name {
		case "lookback":
			if v != math.Trunc(v) {
				return nil, fmt.Errorf("pairs-rv: lookback must be an integral value, got %v", v)
			}
			next.lookback = int(v)
		case "entryZ":
			next.entryZ = v
		case "exitZ":
			next.exitZ = v
		default:
			return nil, fmt.Errorf("pairs-rv: unknown parameter %q", name)
		}
	}

	if next.lookback < 10 {
		return nil, fmt.Errorf("pairs-rv: lookback must be >= 10, got %d", next.lookback)
	}
	if next.exitZ <= 0 {
		return nil, fmt.Errorf("pairs-rv: exitZ must be positive, got %v", next.exitZ)
	}
	if next.exitZ >= next.entryZ {
		return nil, fmt.Errorf("pairs-rv: exitZ (%v) must be less than entryZ (%v)", next.exitZ, next.entryZ)
	}

	return next, nil
}

var _ strategy.Tunable = (*pairsRV)(nil)
var _ strategy.MultiSymbol = (*pairsRV)(nil)

func init() {
	strategy.Register(func() strategy.Strategy { return newPairsRV() })
}
