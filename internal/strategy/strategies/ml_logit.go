package strategies

import (
	"fmt"
	"math"

	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// trials: counted mechanically per walk-forward run — grid is trainWindow x
// enterThresh (2x3 = 6 combos/fold); refitEvery is fixed at 21 (~1 trading
// month), the strategy's definition rather than a fit parameter, the same
// convention as vol-target's checkEvery.

// mlLogitFeatureFloor is the earliest zero-based bar index i for which every
// feature in mlLogitFeatures is computable. The binding constraint is
// volRatio's 63-return standard-deviation window: it needs closes[0..i] to
// contain at least 64 points (63 daily returns), i.e. i >= 63. Every other
// feature (r21, RSI(14), SMA(50)) needs strictly fewer bars.
const mlLogitFeatureFloor = 63

// mlLogitNumFeatures is the width of the feature vector (see
// mlLogitFeatures's doc comment for what each slot holds).
const mlLogitNumFeatures = 6

// mlLogit is a logistic-regression classifier over six technical features,
// predicting the sign of next-bar return and going long when the predicted
// probability of an up move clears a threshold. It is refit periodically on
// a rolling window of labeled examples via full-batch gradient descent — a
// small, from-scratch model chosen specifically so it can be hand-rolled in
// pure Go with no numerical dependency (CLAUDE.md: stdlib only, no new
// dependencies).
//
// NO LOOKAHEAD. A training example for bar i pairs features computed only
// from closes[0..i] with the label sign(close[i+1]/close[i]-1). Given the
// current (completed) bar t — the last element of ctx.History() — the
// newest usable training example is therefore i = t-1: its label needs
// closes[t], which is exactly today's own close, already known to the
// strategy. The model then PREDICTS using features measured at bar t
// itself. No code path in this file ever indexes closes beyond the bar it
// is currently allowed to see.
//
// DETERMINISM. Fitting is full-batch gradient descent with a fixed
// zero-initialized weight vector, a fixed learning rate, a fixed iteration
// count, and examples visited in a fixed (chronological) order — no
// math/rand, no map-iteration-order dependence anywhere in the arithmetic.
// The same bars always produce the same fitted model and the same signals.
//
// RESTART-SAFE CADENCE + memoization. The paper loop runs one bar per fresh
// process, so an in-memory "bars since last fit" counter would never
// persist across sessions — the same failure mode already found live and
// documented on volTarget and rsi2's PositionAge (2026-07-07). The
// governing model is therefore a PURE FUNCTION OF HISTORY LENGTH: it is
// always the model fit at the last refit-cadence boundary,
// fitLen = len(history) - len(history)%refitEvery, on history[:fitLen].
// One long-lived backtest instance and N fresh one-shot paper processes
// reconstruct the identical model for the same bar. (The previous rule —
// "refit when no cached model exists yet" — made refit timing depend on
// process lifetime: a fresh paper process refit every day while the
// backtest refit monthly, so the strategy that traded was not the strategy
// that was validated. Found by independent review 2026-07-15.) The cache
// fields below only memoize the boundary fit so a long-lived instance
// doesn't redundantly refit between boundaries; a cache miss recomputes
// byte-identical weights. This is the one exception to
// "derive-don't-shadow" used throughout this package: it caches a computed
// ARTIFACT (a fitted model), never a position or an expressed intent, so it
// can never desynchronize from the portfolio the way a shadowed position
// counter would.
type mlLogit struct {
	trainWindow int
	enterThresh float64
	refitEvery  int

	// cache: pure-function memoization of the boundary fit (see the type
	// doc comment). cacheFitLen/cacheSym record the history length (always a
	// refit-cadence boundary) and symbol the cached model was fitted at;
	// cacheFitLen == 0 means no model yet.
	cacheFitLen int
	cacheSym    string
	weights     [mlLogitNumFeatures]float64
	bias        float64
	trainMean   [mlLogitNumFeatures]float64
	trainStd    [mlLogitNumFeatures]float64
}

func newMLLogit() *mlLogit {
	return &mlLogit{
		trainWindow: 252,
		enterThresh: 0.55,
		refitEvery:  21, // ~1 trading month; the strategy's definition, not a fit parameter
	}
}

func (s *mlLogit) Name() string { return "ml-logit" }

func (s *mlLogit) Description() string {
	return "Logistic regression on technical features (returns, RSI, vol ratio, SMA distance) predicting next-bar direction; long when P(up) clears a threshold. Refit monthly on a rolling window."
}

func (s *mlLogit) Horizon() strategy.Horizon { return strategy.Short }

// WarmupBars needs the training window PLUS mlLogitFeatureFloor bars before
// it — and then rounds up so that on the engine's first OnBar call
// (history length warmup+1, per the documented convention — see
// donchian.go's WarmupBars comment) the last refit-cadence boundary is
// already deep enough to fit on: the governing model always trains on
// history[:fitLen] with fitLen a multiple of refitEvery (see the type doc
// comment), so the first usable boundary is the first multiple of
// refitEvery >= trainWindow + floor + 1.
func (s *mlLogit) WarmupBars() int {
	needed := s.trainWindow + mlLogitFeatureFloor + 1
	firstBoundary := (needed + s.refitEvery - 1) / s.refitEvery * s.refitEvery
	return firstBoundary - 1
}

// mlLogitFeatures computes the six-element feature vector at bar i from
// closes[0..i] only (no lookahead):
//
//	0: r1       - 1-day return
//	1: r5       - 5-day return
//	2: r21      - 21-day return
//	3: rsi      - RSI(14)/100 - 0.5 (centered)
//	4: volRatio - stdev(returns,10)/stdev(returns,63) - 1; 0 when the
//	              63-return stdev is 0 (a degenerate flat segment), guarded
//	              rather than dividing by zero
//	5: smaDist  - close/SMA(50) - 1
//
// ok is false when bar i doesn't yet reach mlLogitFeatureFloor. Cost is
// O(mlLogitFeatureFloor) regardless of i: only the widest window any
// feature needs (63 returns) is ever materialized.
func mlLogitFeatures(closes []float64, i int) (feat [mlLogitNumFeatures]float64, ok bool) {
	if i < mlLogitFeatureFloor || i >= len(closes) {
		return feat, false
	}

	// Momentum-ratio denominators get the same zero-guard the vol window
	// and SMA already have: one 0.0 close would otherwise send an Inf
	// through standardization and NaN the whole fitted model.
	if closes[i-1] == 0 || closes[i-5] == 0 || closes[i-21] == 0 {
		return feat, false
	}
	feat[0] = closes[i]/closes[i-1] - 1
	feat[1] = closes[i]/closes[i-5] - 1
	feat[2] = closes[i]/closes[i-21] - 1

	rsi, okRSI := strategy.RSI(closes[:i+1], 14)
	if !okRSI {
		return feat, false
	}
	feat[3] = rsi/100 - 0.5

	// The 63-return window ending at i; strategy.StdDev always reads the
	// last n elements of the slice it's given, so this one 63-length window
	// also serves the 10-return stdev — no need to materialize it twice.
	const volWindow = 63
	rets := make([]float64, volWindow)
	for k := 0; k < volWindow; k++ {
		day := i - volWindow + 1 + k
		if closes[day-1] == 0 {
			return feat, false
		}
		rets[k] = closes[day]/closes[day-1] - 1
	}
	sd10, ok10 := strategy.StdDev(rets, 10)
	sd63, ok63 := strategy.StdDev(rets, volWindow)
	if !ok10 || !ok63 {
		return feat, false
	}
	if sd63 == 0 {
		feat[4] = 0
	} else {
		feat[4] = sd10/sd63 - 1
	}

	sma50, okSMA := strategy.SMA(closes[:i+1], 50)
	if !okSMA || sma50 == 0 {
		return feat, false
	}
	feat[5] = closes[i]/sma50 - 1

	return feat, true
}

// sigmoid is the logistic function. math.Exp saturates to +Inf/0 for
// extreme inputs rather than overflowing to NaN, and 1/(1+Inf) = 0,
// 1/(1+0) = 1, so this never returns NaN or Inf.
func sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

// fit trains a fresh model on the rolling window of trainWindow examples
// ending at t-1 (t = len(history)-1, the current bar), storing the result
// in the cache fields on success. ok is false when there isn't enough
// history yet — WarmupBars is set so this should never happen once the
// engine starts calling OnBar, but the check is kept as a defensive
// boundary rather than assumed.
func (s *mlLogit) fit(history []domain.Bar) (ok bool) {
	closes := strategy.Closes(history)
	t := len(closes) - 1

	lo := t - s.trainWindow // earliest example index
	hi := t - 1             // latest example index (label needs closes[hi+1] = closes[t])
	if lo < mlLogitFeatureFloor || hi < lo {
		return false
	}
	n := hi - lo + 1

	xs := make([][mlLogitNumFeatures]float64, n)
	ys := make([]float64, n)
	for k := 0; k < n; k++ {
		i := lo + k
		feat, featOK := mlLogitFeatures(closes, i)
		if !featOK {
			return false
		}
		xs[k] = feat
		if closes[i+1] > closes[i] {
			ys[k] = 1
		} else {
			ys[k] = 0
		}
	}

	// Standardize using ONLY this training window's own statistics — never
	// stats that reach outside [lo, hi], so nothing about the future (or
	// about a later refit's window) leaks into today's model.
	var mean, std [mlLogitNumFeatures]float64
	for j := 0; j < mlLogitNumFeatures; j++ {
		var sum float64
		for k := 0; k < n; k++ {
			sum += xs[k][j]
		}
		m := sum / float64(n)
		var sq float64
		for k := 0; k < n; k++ {
			d := xs[k][j] - m
			sq += d * d
		}
		sd := math.Sqrt(sq / float64(n))
		mean[j] = m
		std[j] = sd
		for k := 0; k < n; k++ {
			if sd > 0 {
				xs[k][j] = (xs[k][j] - m) / sd
			} else {
				xs[k][j] = 0 // guard: zero-variance feature carries no signal this window
			}
		}
	}

	const (
		lr     = 0.1
		lambda = 1e-4
		iters  = 200
	)
	var weights [mlLogitNumFeatures]float64
	var bias float64
	for iter := 0; iter < iters; iter++ {
		var gradW [mlLogitNumFeatures]float64
		var gradB float64
		for k := 0; k < n; k++ {
			z := bias
			for j := 0; j < mlLogitNumFeatures; j++ {
				z += weights[j] * xs[k][j]
			}
			p := sigmoid(z)
			errVal := p - ys[k]
			for j := 0; j < mlLogitNumFeatures; j++ {
				gradW[j] += errVal * xs[k][j]
			}
			gradB += errVal
		}
		for j := 0; j < mlLogitNumFeatures; j++ {
			gradW[j] = gradW[j]/float64(n) + lambda*weights[j]
			weights[j] -= lr * gradW[j]
		}
		bias -= lr * (gradB / float64(n))
	}

	s.weights = weights
	s.bias = bias
	s.trainMean = mean
	s.trainStd = std
	return true
}

// TargetWeights is the pure-history brain: it (re)fits the model per the
// cadence documented on the type, then reports {symbol: 1.0} when the
// fitted model's predicted probability of an up move at the current bar
// clears enterThresh, or {} otherwise. It never reads ctx.Portfolio() — the
// ensemble allocator may call this with virtual books that are none of this
// strategy's business.
func (s *mlLogit) TargetWeights(ctx *strategy.Context) map[string]float64 {
	history := ctx.History()
	if len(history) == 0 {
		return map[string]float64{}
	}
	sym := history[len(history)-1].Symbol

	// The governing model is the one fit at the last refit-cadence
	// boundary — a pure function of history length, identical whether this
	// instance has lived through the whole series or was constructed one
	// bar ago (see the type doc comment). The cache only skips redundant
	// refits at an unchanged boundary.
	fitLen := len(history) - len(history)%s.refitEvery
	if s.trainWindow+mlLogitFeatureFloor+1 > fitLen {
		return map[string]float64{}
	}
	if s.cacheFitLen != fitLen || s.cacheSym != sym {
		if !s.fit(history[:fitLen]) {
			s.cacheFitLen = 0
			return map[string]float64{}
		}
		s.cacheFitLen = fitLen
		s.cacheSym = sym
	}

	closes := strategy.Closes(history)
	t := len(closes) - 1
	feat, ok := mlLogitFeatures(closes, t)
	if !ok {
		return map[string]float64{}
	}

	var x [mlLogitNumFeatures]float64
	for j := 0; j < mlLogitNumFeatures; j++ {
		if s.trainStd[j] > 0 {
			x[j] = (feat[j] - s.trainMean[j]) / s.trainStd[j]
		}
	}
	z := s.bias
	for j := 0; j < mlLogitNumFeatures; j++ {
		z += s.weights[j] * x[j]
	}
	p := sigmoid(z)

	if p >= s.enterThresh {
		return map[string]float64{sym: 1.0}
	}
	return map[string]float64{}
}

// OnBar derives the edge-triggered signal from TargetWeights (the
// pure-history brain) and the ACTUAL portfolio position — never a
// remembered "last signal sent" — so a rejected or unfilled order can never
// leave the strategy believing something the book doesn't (the same
// derive-don't-shadow discipline as every other strategy in this package).
func (s *mlLogit) OnBar(ctx *strategy.Context, bar domain.Bar) []domain.Signal {
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
// over: 2 training-window lengths x 3 entry thresholds (6 combos; no
// cross-parameter constraint excludes any of them).
func (s *mlLogit) ParamSpace() []strategy.ParamDef {
	return []strategy.ParamDef{
		{Name: "trainWindow", Values: []float64{252, 504}},
		{Name: "enterThresh", Values: []float64{0.53, 0.55, 0.57}},
	}
}

// WithParams returns a fresh *mlLogit starting from the defaults, overridden
// by any trainWindow/enterThresh entries in params. It never mutates the
// receiver. Unknown keys, non-integral trainWindow, trainWindow too small to
// clear mlLogitFeatureFloor, and enterThresh outside (0.5,1) are all
// rejected.
//
// refitEvery (the refit cadence) is deliberately NOT tunable — it is part of
// this strategy's definition (how often the model is retrained), the same
// convention as vol-target's checkEvery and dual-momentum's checkEvery, not
// a curve-fitted parameter.
func (s *mlLogit) WithParams(params map[string]float64) (strategy.Strategy, error) {
	next := newMLLogit()

	for name, v := range params {
		switch name {
		case "trainWindow":
			if v != math.Trunc(v) {
				return nil, fmt.Errorf("ml-logit: trainWindow must be an integral value, got %v", v)
			}
			next.trainWindow = int(v)
		case "enterThresh":
			next.enterThresh = v
		default:
			return nil, fmt.Errorf("ml-logit: unknown parameter %q", name)
		}
	}

	if next.trainWindow <= mlLogitFeatureFloor {
		return nil, fmt.Errorf("ml-logit: trainWindow must be > %d, got %d", mlLogitFeatureFloor, next.trainWindow)
	}
	if next.enterThresh <= 0.5 || next.enterThresh >= 1 {
		return nil, fmt.Errorf("ml-logit: enterThresh must be in (0.5,1), got %v", next.enterThresh)
	}

	return next, nil
}

var _ strategy.Tunable = (*mlLogit)(nil)
var _ strategy.TargetWeighter = (*mlLogit)(nil)

func init() {
	strategy.Register(func() strategy.Strategy { return newMLLogit() })
}
