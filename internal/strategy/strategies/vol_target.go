package strategies

import (
	"fmt"
	"math"

	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// trials: counted mechanically per walk-forward run — grid is targetVol x
// lookback (3x2 = 6 combos/fold); checkEvery is fixed, not tunable (cadence
// is the strategy's definition, same convention as dual-momentum's
// checkEvery).

// volTarget is a constant-volatility equity exposure strategy ("risk parity
// lite" per docs/STRATEGIES.md): it holds a single symbol (SPY by
// convention, but the symbol is whatever the engine feeds it) at a weight
// that scales inversely with realized volatility, so the position's
// contribution to portfolio volatility stays roughly constant regardless of
// the underlying's regime.
//
// Derive-don't-shadow: the strategy holds no mutable position-weight state.
// The desired weight w is recomputed fresh from history on every checked
// bar, and the CURRENT weight (wCurrent) is derived from the actual
// portfolio (qty*close/equity), never from a remembered "last signal sent" —
// so a rejected or partially-filled order can never leave the strategy
// believing something the book doesn't. The rebalance cadence is ALSO
// derived rather than kept as process-memory state: an in-memory barsSeen
// counter is restart-unsafe — the paper loop calls OnBar exactly once per
// daily session in a fresh process, so a counter starting at 0 every call
// would never reach checkEvery and the strategy could never trade in paper
// (found live 2026-07-07). Instead cadence is read from history length,
// which is a pure function of the data: in the backtest engine it advances
// exactly once per tick, so the check is equivalent to the old counter, only
// phase-shifted. A flat-book bootstrap (rebalance immediately when the
// symbol is unheld) additionally means the strategy sizes into a position
// right away instead of waiting up to a month for its first entry — true in
// backtest and paper alike. Nothing on the struct is shadowed state anymore.
type volTarget struct {
	targetVol  float64
	lookback   int
	checkEvery int
}

func newVolTarget() *volTarget {
	return &volTarget{
		targetVol:  0.10,
		lookback:   20,
		checkEvery: 21, // ~1 trading month; the strategy's definition, not a fit parameter
	}
}

func (s *volTarget) Name() string { return "vol-target" }

func (s *volTarget) Description() string {
	return "Constant-volatility equity exposure: scale weight inversely with realized volatility, rebalanced monthly."
}

func (s *volTarget) Horizon() strategy.Horizon { return strategy.Long }

// WarmupBars needs lookback+1 closes to compute lookback daily returns.
func (s *volTarget) WarmupBars() int { return s.lookback + 1 }

// desiredWeight computes w = clamp(targetVol / realizedVol, 0, 1), where
// realizedVol is the annualized (sqrt(252) * sample stdev) close-to-close
// return over the last `lookback` returns. ok is false when there isn't
// enough history yet.
func (s *volTarget) desiredWeight(closes []float64) (float64, bool) {
	if len(closes) < s.lookback+1 {
		return 0, false
	}

	window := closes[len(closes)-(s.lookback+1):]
	rets := make([]float64, 0, s.lookback)
	for i := 1; i < len(window); i++ {
		if window[i-1] == 0 {
			continue
		}
		rets = append(rets, window[i]/window[i-1]-1)
	}
	if len(rets) < 2 {
		return 0, false
	}

	sd, ok := strategy.StdDev(rets, len(rets))
	if !ok {
		return 0, false
	}
	realizedVol := sd * math.Sqrt(252)

	if realizedVol == 0 {
		// Degenerate (flat) series: no volatility to size against.
		return 0, true
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

func (s *volTarget) OnBar(ctx *strategy.Context, bar domain.Bar) []domain.Signal {
	// Cadence gate derived from data + portfolio, not process memory — see
	// the struct doc comment. histLen mirrors the old barsSeen%checkEvery
	// check but keyed to data instead of call count. flat bootstraps a
	// strategy holding nothing straight into evaluation.
	histLen := len(ctx.History())
	cadenceDue := histLen%s.checkEvery == 0
	flat := true
	if pos, ok := ctx.Portfolio().Positions[bar.Symbol]; ok && pos.Qty != 0 {
		flat = false
	}
	if !cadenceDue && !flat {
		return nil
	}

	closes := strategy.Closes(ctx.History())
	w, ok := s.desiredWeight(closes)
	if !ok {
		return nil
	}

	// Current weight, derived from the actual portfolio — never from a
	// remembered target. Flat (no position, or missing) is weight 0.
	wCurrent := 0.0
	if pos, ok := ctx.Portfolio().Positions[bar.Symbol]; ok && pos.Qty != 0 {
		equity := ctx.Portfolio().Equity(map[string]float64{bar.Symbol: bar.Close})
		if equity != 0 {
			wCurrent = float64(pos.Qty) * bar.Close / equity
		}
	}

	if math.Abs(w-wCurrent) <= 0.05 {
		return nil
	}

	return []domain.Signal{{Symbol: bar.Symbol, TargetWeight: w}}
}

// ParamSpace declares the discrete grid walk-forward optimization searches
// over: 3 target-vol levels x 2 lookback windows (6 combos; no cross-param
// constraint excludes any of them).
func (s *volTarget) ParamSpace() []strategy.ParamDef {
	return []strategy.ParamDef{
		{Name: "targetVol", Values: []float64{0.08, 0.10, 0.12}},
		{Name: "lookback", Values: []float64{20, 60}},
	}
}

// WithParams returns a fresh *volTarget starting from the defaults,
// overridden by any targetVol/lookback entries in params. It never mutates
// the receiver. Unknown keys, targetVol outside (0,1), non-integral
// lookback, and lookback < 5 are all rejected.
//
// checkEvery (the rebalance-check cadence) is deliberately NOT tunable, the
// same convention as dual-momentum's checkEvery: it is part of this
// strategy's definition (how often exposure is re-evaluated), not a
// curve-fitted parameter.
func (s *volTarget) WithParams(params map[string]float64) (strategy.Strategy, error) {
	next := newVolTarget()

	for name, v := range params {
		switch name {
		case "targetVol":
			next.targetVol = v
		case "lookback":
			if v != math.Trunc(v) {
				return nil, fmt.Errorf("vol-target: lookback must be an integral value, got %v", v)
			}
			next.lookback = int(v)
		default:
			return nil, fmt.Errorf("vol-target: unknown parameter %q", name)
		}
	}

	if next.targetVol <= 0 || next.targetVol >= 1 {
		return nil, fmt.Errorf("vol-target: targetVol must be in (0,1), got %v", next.targetVol)
	}
	if next.lookback < 5 {
		return nil, fmt.Errorf("vol-target: lookback must be >= 5, got %d", next.lookback)
	}

	return next, nil
}

var _ strategy.Tunable = (*volTarget)(nil)

func init() {
	strategy.Register(func() strategy.Strategy { return newVolTarget() })
}
