package strategies

import (
	"fmt"
	"math"

	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// trials: counted mechanically per walk-forward run — grid is entryRSI x
// exitRSI x timeStop (3x3x2 = 18 combos/fold); WithParams's cross-param
// validation (entryRSI < exitRSI, etc.) excludes invalid combos from the
// trial count.

// rsi2 is Larry Connors' RSI(2) mean-reversion strategy: buy oversold dips
// within an established uptrend, and exit quickly. Entry: Close > SMA200 and
// RSI(2) < 10 -> long at entryWeight. Exit: RSI(2) > 60, or after 10 bars in
// the position (time stop), whichever comes first.
//
// ALL position state is DERIVED from the environment, never shadowed in
// strategy fields: whether we are "in a position" comes from the portfolio
// (ctx.Portfolio()), and how long we have held comes from
// ctx.PositionAge() — supplied by the backtest engine's fill log or the
// paper session's persisted order ledger. An in-memory counter would
// silently reset every process restart (each daily paper session is a
// fresh process), so the time stop could never fire in paper trading.
// The struct therefore holds parameters only — no mutable state.
type rsi2 struct {
	smaPeriod   int
	rsiPeriod   int
	entryRSI    float64
	exitRSI     float64
	timeStop    int
	entryWeight float64
}

func newRSI2() *rsi2 {
	return &rsi2{
		smaPeriod: 200,
		rsiPeriod: 2,
		entryRSI:  10,
		exitRSI:   60,
		timeStop:  10,
		// Sizing authority is the risk manager's: MaxPositionWeight is 0.20,
		// so requesting more would only be clamped down. Ask for exactly
		// what the risk manager will allow.
		entryWeight: 0.20,
	}
}

func (s *rsi2) Name() string { return "rsi2" }

func (s *rsi2) Description() string {
	return "Connors RSI(2) mean reversion: buy oversold dips above SMA200, exit on RSI(2)>60 or a 10-bar time stop."
}

func (s *rsi2) Horizon() strategy.Horizon { return strategy.Short }

func (s *rsi2) WarmupBars() int { return 200 }

func (s *rsi2) OnBar(ctx *strategy.Context, bar domain.Bar) []domain.Signal {
	closes := strategy.Closes(ctx.History())

	sma200, okSMA := strategy.SMA(closes, s.smaPeriod)
	rsi, okRSI := strategy.RSI(closes, s.rsiPeriod)
	if !okSMA || !okRSI {
		return nil
	}

	// Derive position state from the actual portfolio, not from whether we
	// once emitted an entry signal.
	qty := int64(0)
	if pos, ok := ctx.Portfolio().Positions[bar.Symbol]; ok {
		qty = pos.Qty
	}
	inPosition := qty > 0

	if inPosition {
		// Derived position age (counts the current bar; first in-position
		// OnBar sees 1). age == 0 while holding means "age unknown" — the
		// safe-degradation convention: treat the position as just-entered
		// and let only the RSI exit fire, never the time stop.
		age := ctx.PositionAge(bar.Symbol)
		if rsi > s.exitRSI || (age > 0 && age >= s.timeStop) {
			return []domain.Signal{{Symbol: bar.Symbol, TargetWeight: 0.0}}
		}
		return nil
	}

	if bar.Close > sma200 && rsi < s.entryRSI {
		return []domain.Signal{{Symbol: bar.Symbol, TargetWeight: s.entryWeight}}
	}

	return nil
}

// ParamSpace declares the discrete grid walk-forward optimization searches
// over: 3 entry-RSI thresholds x 3 exit-RSI thresholds x 2 time stops (18
// combos, minus any invalid combination WithParams rejects).
func (s *rsi2) ParamSpace() []strategy.ParamDef {
	return []strategy.ParamDef{
		{Name: "entryRSI", Values: []float64{5, 10, 15}},
		{Name: "exitRSI", Values: []float64{60, 70, 80}},
		{Name: "timeStop", Values: []float64{5, 10}},
	}
}

// WithParams returns a fresh *rsi2 starting from the defaults, overridden by
// any entryRSI/exitRSI/timeStop entries in params. It never mutates the
// receiver. Unknown keys, entryRSI <= 0, exitRSI >= 100, entryRSI >=
// exitRSI, non-integral timeStop, and timeStop < 1 are all rejected.
func (s *rsi2) WithParams(params map[string]float64) (strategy.Strategy, error) {
	next := newRSI2()

	for name, v := range params {
		switch name {
		case "entryRSI":
			next.entryRSI = v
		case "exitRSI":
			next.exitRSI = v
		case "timeStop":
			if v != math.Trunc(v) {
				return nil, fmt.Errorf("rsi2: timeStop must be an integral value, got %v", v)
			}
			next.timeStop = int(v)
		default:
			return nil, fmt.Errorf("rsi2: unknown parameter %q", name)
		}
	}

	if next.entryRSI <= 0 {
		return nil, fmt.Errorf("rsi2: entryRSI must be positive, got %v", next.entryRSI)
	}
	if next.exitRSI >= 100 {
		return nil, fmt.Errorf("rsi2: exitRSI must be < 100, got %v", next.exitRSI)
	}
	if next.entryRSI >= next.exitRSI {
		return nil, fmt.Errorf("rsi2: entryRSI (%v) must be less than exitRSI (%v)", next.entryRSI, next.exitRSI)
	}
	if next.timeStop < 1 {
		return nil, fmt.Errorf("rsi2: timeStop must be >= 1, got %d", next.timeStop)
	}

	return next, nil
}

var _ strategy.Tunable = (*rsi2)(nil)

func init() {
	strategy.Register(func() strategy.Strategy { return newRSI2() })
}
