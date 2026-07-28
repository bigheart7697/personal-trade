package strategies

import (
	"fmt"
	"math"

	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// trials: counted mechanically per walk-forward run — grid is lookback x
// targetVol (2x2 = 4 combos/fold); skip is fixed at 21, not tunable (it is
// part of the 12-1 convention's definition, the same convention as
// vol-target's checkEvery), so WithParams never varies it and it is excluded
// from ParamSpace.

// tsmomSkip is the fixed "skip the most recent month" gap in the classic
// 12-1 time-series momentum convention: momentum is measured from
// lookback bars ago to skip bars ago, deliberately excluding the most
// recent month to avoid the well-documented short-term reversal effect that
// contaminates a naive 12-0 momentum signal. This is part of the strategy's
// definition, not a fit parameter (see ParamSpace's doc comment).
const tsmomSkip = 21

// tsmomVolWindow is the fixed number of trailing daily returns used to
// estimate realized volatility for constant-vol sizing.
const tsmomVolWindow = 63

// tsmomCadence is the fixed rebalance cadence (~1 trading month), the same
// convention as vol-target's/dual-momentum's checkEvery: part of the
// strategy's definition, not tunable.
const tsmomCadence = 21

// tsmom is 12-1 time-series momentum (Moskowitz/Ooi/Pedersen-style): long a
// single symbol whenever its trailing-year return (skipping the most recent
// month) is positive, sized to a constant volatility target rather than a
// fixed weight, so a position in a calm regime is larger than the same
// signal in a turbulent one. Rebalanced monthly.
//
// Known weaknesses, documented rather than hidden: (1) it is long-only and
// single-symbol here — the classic academic implementation trades a
// diversified futures/asset-class universe and can go short; a long-only
// equity version captures only half the original edge and inherits full
// buy-and-hold drawdown risk whenever momentum stays positive through a
// crash's early stages (autocorrelation breaks down exactly when it would
// help most). (2) The monthly rebalance means the vol-scaling can lag a
// sudden vol regime shift by up to one cadence cycle.
//
// Derive-don't-shadow, TargetWeighter seam: the struct holds parameters
// only. TargetWeights is a PURE function of history (see its own doc
// comment for how the monthly cadence is derived without any stored state
// or ctx.Portfolio() read), and OnBar diffs that pure target against the
// ACTUAL portfolio-derived current weight to decide what to signal — so a
// rejected or unfilled order can never leave the strategy believing
// something the book doesn't, and a fresh paper-loop process reproduces
// exactly what a long-running backtest would have computed at the same
// history length (the same restart-safety fix already applied to
// vol-target's checkEvery and dual-momentum's checkEvery, found live
// 2026-07-07).
type tsmom struct {
	lookback  int
	targetVol float64
}

func newTSMOM() *tsmom {
	return &tsmom{lookback: 252, targetVol: 0.10}
}

func (s *tsmom) Name() string { return "tsmom" }

func (s *tsmom) Description() string {
	return "12-1 time-series momentum: long when the trailing year (minus the last month) is up, sized to a constant volatility target. Rebalances monthly."
}

func (s *tsmom) Horizon() strategy.Horizon { return strategy.Long }

// WarmupBars needs lookback bars for the momentum measurement plus
// tsmomVolWindow(63)+1 bars for the volatility estimate; lookback (>=126 in
// the grid) already dominates the vol window, so lookback+64 is comfortably
// enough for both (mirrors the existing per-file convention of stating the
// arithmetic rather than just the number).
func (s *tsmom) WarmupBars() int { return s.lookback + 64 }

// momentumAndWeight computes tsmom's raw (non-cadence) signal from a
// history slice, as a pure function of that slice: mom = close[n-1-skip] /
// close[n-1-lookback] - 1 (n = len(history)), and, when mom > 0, a
// constant-volatility position size w = clamp(targetVol/realizedVol, 0, 1),
// where realizedVol is the annualized (sqrt(252) * sample stdev) daily
// return over the last tsmomVolWindow returns.
//
// ok is false only when history is too short to compute momentum at all. A
// mom <= 0, or a too-short/zero-vol case once mom > 0, both legitimately
// return (0, true) — "flat, computed", not "unknown" — the guard the spec
// asks for against dividing by (or sizing against) zero/short history.
func (s *tsmom) momentumAndWeight(history []domain.Bar) (float64, bool) {
	n := len(history)
	if n < s.lookback+1 {
		return 0, false
	}

	closes := strategy.Closes(history)
	closeSkip := closes[n-1-tsmomSkip]
	closePast := closes[n-1-s.lookback]
	if closePast == 0 {
		return 0, false
	}
	mom := closeSkip/closePast - 1
	if mom <= 0 {
		return 0, true
	}

	if n < tsmomVolWindow+1 {
		return 0, true // guard short history: can't size a vol target, stay flat
	}
	window := closes[n-(tsmomVolWindow+1):]
	rets := make([]float64, 0, tsmomVolWindow)
	for i := 1; i < len(window); i++ {
		if window[i-1] == 0 {
			continue
		}
		rets = append(rets, window[i]/window[i-1]-1)
	}
	if len(rets) < 2 {
		return 0, true
	}
	sd, ok := strategy.StdDev(rets, len(rets))
	if !ok {
		return 0, true
	}
	realizedVol := sd * math.Sqrt(252)
	if realizedVol == 0 {
		return 0, true // guard zero vol: degenerate series, nothing to size against
	}

	w := s.targetVol / realizedVol
	if w < 0 {
		w = 0
	}
	if w > 1 {
		w = 1
	}
	return w, true
}

// TargetWeights answers "what does tsmom want right now" as a pure function
// of price history (the strategy.TargetWeighter seam) — it must not, and
// does not, read ctx.Portfolio().
//
// Rebalance cadence: the target is normally locked to the value computed at
// the LAST cadence tick (the largest multiple of tsmomCadence bars <= the
// current history length), recomputed deterministically from the truncated
// history at that tick every time — never cached in the struct, so a fresh
// process (paper loop) reproduces a long-running backtest's answer exactly
// at the same history length. Between ticks the answer is therefore stable,
// matching the spec's "recompute target only when len(history) % 21 == 0".
//
// Bootstrap: if the last tick's answer was flat (0) but today's raw,
// non-cadence momentum is positive, this returns today's value instead of
// waiting up to a month for the next tick boundary — the same flat-book
// bootstrap idea used by vol-target/dual-momentum's OnBar (found live
// 2026-07-07), reinterpreted purely in terms of the strategy's OWN
// last-derived intent (pure history) rather than ctx.Portfolio(), since a
// TargetWeighter must never read the portfolio.
func (s *tsmom) TargetWeights(ctx *strategy.Context) map[string]float64 {
	history := ctx.History()
	if len(history) == 0 {
		return map[string]float64{}
	}
	sym := history[len(history)-1].Symbol

	n := len(history)
	lastTick := (n / tsmomCadence) * tsmomCadence // last cadence tick <= n; 0 before the first

	tickWeight := 0.0
	if lastTick > 0 {
		if w, ok := s.momentumAndWeight(history[:lastTick]); ok {
			tickWeight = w
		}
	}

	if tickWeight == 0 {
		if w, ok := s.momentumAndWeight(history); ok && w > 0 {
			return map[string]float64{sym: w}
		}
		return map[string]float64{}
	}

	return map[string]float64{sym: tickWeight}
}

// rebalanceBand is the minimum |target - current| weight gap that triggers
// a signal, the same churn-avoidance convention as vol-target's OnBar: the
// CURRENT weight (qty*close/equity) drifts with price even when the target
// hasn't moved, so an exact-equality comparison would resignal on harmless
// day-to-day drift instead of only at real cadence-tick changes.
const tsmomRebalanceBand = 0.05

// OnBar gates trading attempts on the SAME cadence (histLen % tsmomCadence
// == 0) or flat-book bootstrap used by TargetWeights, applied here against
// the actual portfolio (which OnBar, unlike TargetWeights, is allowed to
// read) — the same convention as vol-target's/dual-momentum's OnBar cadence
// gate. Without this gate, tsmom's uncapped vol-scaled weight (often well
// above risk.Manager's MaxPositionWeight, e.g. 0.20) would sit clamped
// there indefinitely: the ACTUAL portfolio weight can never rise to meet
// the raw target, so an ungated diff against the risk-capped current weight
// would exceed tsmomRebalanceBand on nearly every bar and re-signal daily —
// each re-signal a redundant no-op the risk manager just rejects (found
// while smoke-testing against real SPY data: thousands of same-day
// resubmit-and-reject cycles). Gating to the monthly cadence (matching the
// "rebalances monthly" claim in Description) makes the attempt frequency
// match the strategy's own definition instead of firing every single bar.
func (s *tsmom) OnBar(ctx *strategy.Context, bar domain.Bar) []domain.Signal {
	histLen := len(ctx.History())
	cadenceDue := histLen%tsmomCadence == 0
	flat := true
	if pos, ok := ctx.Portfolio().Positions[bar.Symbol]; ok && pos.Qty != 0 {
		flat = false
	}
	if !cadenceDue && !flat {
		return nil
	}

	target := s.TargetWeights(ctx)
	want := target[bar.Symbol] // 0 if absent

	// Current weight, derived from the actual portfolio — never from a
	// remembered target.
	current := 0.0
	if pos, ok := ctx.Portfolio().Positions[bar.Symbol]; ok && pos.Qty != 0 {
		equity := ctx.Portfolio().Equity(map[string]float64{bar.Symbol: bar.Close})
		if equity != 0 {
			current = float64(pos.Qty) * bar.Close / equity
		}
	}

	if math.Abs(want-current) <= tsmomRebalanceBand {
		return nil
	}
	return []domain.Signal{{Symbol: bar.Symbol, TargetWeight: want}}
}

// ParamSpace declares the discrete grid walk-forward optimization searches
// over: 2 lookback windows x 2 target-vol levels (4 combos; no cross-param
// constraint excludes any of them). tsmomSkip and tsmomCadence are
// deliberately NOT tunable — they are part of the 12-1 convention's
// definition, not curve-fitted parameters.
func (s *tsmom) ParamSpace() []strategy.ParamDef {
	return []strategy.ParamDef{
		{Name: "lookback", Values: []float64{126, 252}},
		{Name: "targetVol", Values: []float64{0.10, 0.15}},
	}
}

// WithParams returns a fresh *tsmom starting from the defaults, overridden
// by any lookback/targetVol entries in params. It never mutates the
// receiver. Unknown keys, non-integral lookback, lookback <= tsmomSkip
// (there would be no trailing-year window left after skipping the last
// month), and targetVol outside (0,1) are all rejected.
func (s *tsmom) WithParams(params map[string]float64) (strategy.Strategy, error) {
	next := newTSMOM()

	for name, v := range params {
		switch name {
		case "lookback":
			if v != math.Trunc(v) {
				return nil, fmt.Errorf("tsmom: lookback must be an integral value, got %v", v)
			}
			next.lookback = int(v)
		case "targetVol":
			next.targetVol = v
		default:
			return nil, fmt.Errorf("tsmom: unknown parameter %q", name)
		}
	}

	if next.lookback <= tsmomSkip {
		return nil, fmt.Errorf("tsmom: lookback must be > %d (the fixed skip), got %d", tsmomSkip, next.lookback)
	}
	if next.targetVol <= 0 || next.targetVol >= 1 {
		return nil, fmt.Errorf("tsmom: targetVol must be in (0,1), got %v", next.targetVol)
	}

	return next, nil
}

var _ strategy.Tunable = (*tsmom)(nil)
var _ strategy.TargetWeighter = (*tsmom)(nil)

func init() {
	strategy.Register(func() strategy.Strategy { return newTSMOM() })
}
