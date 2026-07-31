package strategies

import (
	"fmt"
	"math"
	"sort"

	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// trials: counted mechanically per walk-forward run — grid is trainWindow x
// deadband (2x2 = 4 combos/fold); refitEvery is fixed at 63 (~1 trading
// quarter), the strategy's definition rather than a fit parameter, the same
// convention as vol-target's checkEvery and ml-logit's refitEvery.

// mlBoostFeatureFloor mirrors ml-logit's: the earliest zero-based bar index
// i for which every feature in mlBoostFeatures is computable — bound by the
// 63-return volRatio window (see mlLogitFeatureFloor's doc comment; this is
// an independent copy per this package's one-file-per-strategy convention).
const mlBoostFeatureFloor = 63

// mlBoostNumFeatures is the feature vector width: the same six technical
// features as ml-logit (see mlBoostFeatures).
const mlBoostNumFeatures = 6

// mlBoostM is the fixed number of boosting rounds (decision stumps).
const mlBoostM = 50

// mlBoostLR is the fixed boosting learning rate (shrinkage applied to every
// stump's leaf values before adding them to the running score).
const mlBoostLR = 0.1

// mlBoostThresholds is the fixed number of candidate split thresholds tried
// per feature per round.
const mlBoostThresholds = 8

// mlBoostStump is one gradient-boosted decision stump: split on standardized
// feature `feature` at `threshold`. leftValue/rightValue are already scaled
// by mlBoostLR, so scoring a feature vector is a plain sum of stump outputs.
type mlBoostStump struct {
	feature    int
	threshold  float64
	leftValue  float64
	rightValue float64
}

// mlBoost is a gradient-boosted ensemble of decision stumps (hand-rolled,
// pure Go — no numerical dependency, per CLAUDE.md's stdlib-only rule) over
// the same six technical features as ml-logit, but regressing next-bar
// RETURN directly under squared loss rather than classifying its sign. It
// goes long while the boosted score is comfortably positive.
//
// NO LOOKAHEAD: identical invariant to ml-logit (see its doc comment) — a
// training example for bar i uses features from closes[0..i] and the label
// close[i+1]/close[i]-1; closes[i+1] is today's own close when i = t-1 (t =
// the current bar, the last element of ctx.History()). Predictions use
// features measured at bar t itself. WarmupBars enforces that the earliest
// training example never falls before mlBoostFeatureFloor.
//
// DETERMINISM: no randomness anywhere. Each round considers all 6 features x
// 8 fixed quantile thresholds — computed by sorting a COPY of the training
// window's standardized feature values; the xs slice used for residual
// bookkeeping is never reordered — and picks the (feature, threshold)
// minimizing squared error against the current residuals, with a
// deterministic tie-break (lowest feature index first, then lowest
// threshold) implemented simply by scanning features 0..5 and thresholds in
// ascending order and requiring a STRICT improvement to replace the
// incumbent best split.
//
// RESTART-SAFE CADENCE: the same boundary-fit pattern as ml-logit (see its
// doc comment for the full rationale) — the governing model is always the
// one fit at the last refit-cadence boundary,
// fitLen = len(history) - len(history)%refitEvery, on history[:fitLen], a
// pure function of history length that one long-lived backtest instance
// and N fresh one-shot paper processes reconstruct identically. The cache
// fields only memoize the boundary fit; a cache miss recomputes
// byte-identical stumps. This is the one exception to
// "derive-don't-shadow" used throughout this package, because it caches a
// computed model artifact, never a position or an expressed intent.
type mlBoost struct {
	trainWindow int
	deadband    float64
	refitEvery  int

	cacheFitLen int
	cacheSym    string
	stumps      []mlBoostStump
	trainMean   [mlBoostNumFeatures]float64
	trainStd    [mlBoostNumFeatures]float64
}

func newMLBoost() *mlBoost {
	return &mlBoost{
		trainWindow: 504,
		deadband:    0.0,
		refitEvery:  63, // ~1 trading quarter; the strategy's definition, not a fit parameter
	}
}

func (s *mlBoost) Name() string { return "ml-boost" }

func (s *mlBoost) Description() string {
	return "Gradient-boosted decision stumps on technical features predicting next-bar return; long while the boosted score is positive. Refit quarterly on a rolling window."
}

func (s *mlBoost) Horizon() strategy.Horizon { return strategy.Long }

// WarmupBars needs the training window PLUS mlBoostFeatureFloor bars before
// it — rounded up so the engine's first OnBar call (history length
// warmup+1) already has a refit-cadence boundary deep enough to fit on,
// exactly as ml-logit's WarmupBars documents.
func (s *mlBoost) WarmupBars() int {
	needed := s.trainWindow + mlBoostFeatureFloor + 1
	firstBoundary := (needed + s.refitEvery - 1) / s.refitEvery * s.refitEvery
	return firstBoundary - 1
}

// mlBoostFeatures is definitionally identical to ml-logit's feature vector
// (kept as an independent copy per this package's one-file-per-strategy
// convention): r1, r5, r21, rsi, volRatio, smaDist, computed from
// closes[0..i] only. See mlLogitFeatures for the field-by-field
// documentation.
func mlBoostFeatures(closes []float64, i int) (feat [mlBoostNumFeatures]float64, ok bool) {
	if i < mlBoostFeatureFloor || i >= len(closes) {
		return feat, false
	}

	// Same zero-guard as mlLogitFeatures: one 0.0 close would otherwise
	// NaN the whole fitted model through standardization.
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

// fit trains a fresh boosted-stump ensemble on the rolling window of
// trainWindow examples ending at t-1 (t = len(history)-1, the current bar),
// storing the result in the cache fields on success. ok is false when there
// isn't enough history yet (a defensive boundary — WarmupBars should
// prevent this from ever firing once the engine starts calling OnBar).
func (s *mlBoost) fit(history []domain.Bar) (ok bool) {
	closes := strategy.Closes(history)
	t := len(closes) - 1

	lo := t - s.trainWindow
	hi := t - 1
	if lo < mlBoostFeatureFloor || hi < lo {
		return false
	}
	n := hi - lo + 1

	xs := make([][mlBoostNumFeatures]float64, n)
	ys := make([]float64, n)
	for k := 0; k < n; k++ {
		i := lo + k
		feat, featOK := mlBoostFeatures(closes, i)
		if !featOK {
			return false
		}
		xs[k] = feat
		ys[k] = closes[i+1]/closes[i] - 1
	}

	// Standardize using ONLY this training window's own statistics (see
	// ml-logit's fit for the identical rationale).
	var mean, std [mlBoostNumFeatures]float64
	for j := 0; j < mlBoostNumFeatures; j++ {
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
				xs[k][j] = 0
			}
		}
	}

	residual := make([]float64, n)
	copy(residual, ys)

	stumps := make([]mlBoostStump, 0, mlBoostM)
	sortBuf := make([]float64, n)

	for round := 0; round < mlBoostM; round++ {
		bestSSE := math.Inf(1)
		bestFeat := -1
		var bestThr, bestLeft, bestRight float64

		for feat := 0; feat < mlBoostNumFeatures; feat++ {
			for k := 0; k < n; k++ {
				sortBuf[k] = xs[k][feat]
			}
			sort.Float64s(sortBuf) // sorts the COPY only; xs/residual order is untouched

			for tIdx := 1; tIdx <= mlBoostThresholds; tIdx++ {
				frac := float64(tIdx) / float64(mlBoostThresholds+1)
				idx := int(frac * float64(n-1))
				thr := sortBuf[idx]

				var leftSum, rightSum float64
				var leftN, rightN int
				for k := 0; k < n; k++ {
					if xs[k][feat] <= thr {
						leftSum += residual[k]
						leftN++
					} else {
						rightSum += residual[k]
						rightN++
					}
				}
				if leftN == 0 || rightN == 0 {
					continue // degenerate split: every example fell on one side
				}
				leftMean := leftSum / float64(leftN)
				rightMean := rightSum / float64(rightN)

				var sse float64
				for k := 0; k < n; k++ {
					pred := rightMean
					if xs[k][feat] <= thr {
						pred = leftMean
					}
					d := residual[k] - pred
					sse += d * d
				}

				// Strict improvement only: combined with the ascending
				// feature/threshold scan order, this implements the
				// documented tie-break (lowest feature index, then lowest
				// threshold) without any extra bookkeeping.
				if sse < bestSSE {
					bestSSE = sse
					bestFeat = feat
					bestThr = thr
					bestLeft = leftMean
					bestRight = rightMean
				}
			}
		}

		if bestFeat < 0 {
			break // no valid split found this round (degenerate residuals); stop boosting early
		}

		st := mlBoostStump{
			feature:    bestFeat,
			threshold:  bestThr,
			leftValue:  mlBoostLR * bestLeft,
			rightValue: mlBoostLR * bestRight,
		}
		stumps = append(stumps, st)

		for k := 0; k < n; k++ {
			out := st.rightValue
			if xs[k][bestFeat] <= bestThr {
				out = st.leftValue
			}
			residual[k] -= out
		}
	}

	s.stumps = stumps
	s.trainMean = mean
	s.trainStd = std
	return true
}

// mlBoostScore sums every stump's contribution for a standardized feature
// vector.
func mlBoostScore(stumps []mlBoostStump, x [mlBoostNumFeatures]float64) float64 {
	var score float64
	for _, st := range stumps {
		if x[st.feature] <= st.threshold {
			score += st.leftValue
		} else {
			score += st.rightValue
		}
	}
	return score
}

// TargetWeights is the pure-history brain: it (re)fits the model per the
// cadence documented on the type, then reports {symbol: 1.0} when the
// boosted score at the current bar clears deadband, or {} otherwise. It
// never reads ctx.Portfolio().
func (s *mlBoost) TargetWeights(ctx *strategy.Context) map[string]float64 {
	history := ctx.History()
	if len(history) == 0 {
		return map[string]float64{}
	}
	sym := history[len(history)-1].Symbol

	// Boundary fit — pure function of history length; see the type doc
	// comment and ml-logit's TargetWeights for the full rationale.
	fitLen := len(history) - len(history)%s.refitEvery
	if s.trainWindow+mlBoostFeatureFloor+1 > fitLen {
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
	feat, ok := mlBoostFeatures(closes, t)
	if !ok {
		return map[string]float64{}
	}

	var x [mlBoostNumFeatures]float64
	for j := 0; j < mlBoostNumFeatures; j++ {
		if s.trainStd[j] > 0 {
			x[j] = (feat[j] - s.trainMean[j]) / s.trainStd[j]
		}
	}
	score := mlBoostScore(s.stumps, x)

	// A deadband on the score, rather than any dependence on the current
	// position: TargetWeights must stay a pure function of history, so
	// churn-avoidance can't read ctx.Portfolio() here. Some extra
	// round-trip churn near the deadband is an accepted tradeoff (per the
	// design brief).
	if score > s.deadband {
		return map[string]float64{sym: 1.0}
	}
	return map[string]float64{}
}

// OnBar derives the edge-triggered signal from TargetWeights (the
// pure-history brain) and the ACTUAL portfolio position — never a
// remembered "last signal sent" (the same derive-don't-shadow discipline as
// every other strategy in this package).
func (s *mlBoost) OnBar(ctx *strategy.Context, bar domain.Bar) []domain.Signal {
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
// over: 2 training-window lengths x 2 deadbands (4 combos; no
// cross-parameter constraint excludes any of them).
func (s *mlBoost) ParamSpace() []strategy.ParamDef {
	return []strategy.ParamDef{
		{Name: "trainWindow", Values: []float64{504, 756}},
		{Name: "deadband", Values: []float64{0.0, 0.0005}},
	}
}

// WithParams returns a fresh *mlBoost starting from the defaults, overridden
// by any trainWindow/deadband entries in params. It never mutates the
// receiver. Unknown keys, non-integral trainWindow, trainWindow too small to
// clear mlBoostFeatureFloor, and a negative deadband are all rejected.
//
// refitEvery is deliberately NOT tunable — the same convention as
// ml-logit's refitEvery and vol-target's checkEvery: it is part of this
// strategy's definition, not a curve-fitted parameter.
func (s *mlBoost) WithParams(params map[string]float64) (strategy.Strategy, error) {
	next := newMLBoost()

	for name, v := range params {
		switch name {
		case "trainWindow":
			if v != math.Trunc(v) {
				return nil, fmt.Errorf("ml-boost: trainWindow must be an integral value, got %v", v)
			}
			next.trainWindow = int(v)
		case "deadband":
			next.deadband = v
		default:
			return nil, fmt.Errorf("ml-boost: unknown parameter %q", name)
		}
	}

	if next.trainWindow <= mlBoostFeatureFloor {
		return nil, fmt.Errorf("ml-boost: trainWindow must be > %d, got %d", mlBoostFeatureFloor, next.trainWindow)
	}
	if next.deadband < 0 {
		return nil, fmt.Errorf("ml-boost: deadband must be >= 0, got %v", next.deadband)
	}

	return next, nil
}

var _ strategy.Tunable = (*mlBoost)(nil)
var _ strategy.TargetWeighter = (*mlBoost)(nil)

func init() {
	strategy.Register(func() strategy.Strategy { return newMLBoost() })
}
