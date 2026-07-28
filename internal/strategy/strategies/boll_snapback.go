package strategies

import (
	"fmt"
	"math"

	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// trials: counted mechanically per walk-forward run — grid is period x k x
// timeStop (2x3x2 = 12 combos/fold); WithParams has no cross-param
// constraint here (period, k, timeStop are independent), so all 12 combos
// are valid.

// bollSnapback is a Bollinger Band mean-reversion strategy: buy a close that
// has dropped k standard deviations below its SMA while the market remains
// in an established uptrend (close > SMA200 regime filter), and exit as
// soon as price snaps back to the mean or a time stop elapses.
//
// ALL position state is DERIVED from the environment, never shadowed in
// strategy fields: whether we are "in a position" comes from the portfolio
// (ctx.Portfolio()), and how long we have held comes from
// ctx.PositionAge() — supplied by the backtest engine's fill log or the
// paper session's persisted order ledger. An in-memory counter would
// silently reset every process restart (each daily paper session is a
// fresh process), so the time stop could never fire in paper trading.
// The struct therefore holds parameters only — no mutable state.
type bollSnapback struct {
	smaPeriod   int
	period      int
	k           float64
	timeStop    int
	entryWeight float64
}

func newBollSnapback() *bollSnapback {
	return &bollSnapback{
		smaPeriod: 200,
		period:    20,
		k:         2.0,
		timeStop:  10,
		// Sizing authority is the risk manager's: MaxPositionWeight is 0.20,
		// so requesting more would only be clamped down. Ask for exactly
		// what the risk manager will allow.
		entryWeight: 0.20,
	}
}

func (s *bollSnapback) Name() string { return "boll-snapback" }

func (s *bollSnapback) Description() string {
	return "Bollinger snap-back mean reversion: buy closes k stdevs below the band's SMA above SMA200, exit on mean touch or a time stop."
}

func (s *bollSnapback) Horizon() strategy.Horizon { return strategy.Short }

// WarmupBars is dominated by the SMA200 regime filter: it is always the
// widest window this strategy needs, regardless of the period/k/timeStop
// parameters.
func (s *bollSnapback) WarmupBars() int { return s.smaPeriod }

func (s *bollSnapback) OnBar(ctx *strategy.Context, bar domain.Bar) []domain.Signal {
	closes := strategy.Closes(ctx.History())

	sma200, okSMA200 := strategy.SMA(closes, s.smaPeriod)
	sma, okSMA := strategy.SMA(closes, s.period)
	sd, okSD := strategy.StdDev(closes, s.period)
	if !okSMA200 || !okSMA || !okSD {
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
		// and let only the mean-touch exit fire, never the time stop.
		age := ctx.PositionAge(bar.Symbol)
		if bar.Close >= sma || (age > 0 && age >= s.timeStop) {
			return []domain.Signal{{Symbol: bar.Symbol, TargetWeight: 0.0}}
		}
		return nil
	}

	lowerBand := sma - s.k*sd
	if bar.Close > sma200 && bar.Close < lowerBand {
		return []domain.Signal{{Symbol: bar.Symbol, TargetWeight: s.entryWeight}}
	}

	return nil
}

// ParamSpace declares the discrete grid walk-forward optimization searches
// over: 2 band periods x 3 band widths x 2 time stops (12 combos; no
// cross-param constraint excludes any of them).
func (s *bollSnapback) ParamSpace() []strategy.ParamDef {
	return []strategy.ParamDef{
		{Name: "period", Values: []float64{10, 20}},
		{Name: "k", Values: []float64{1.5, 2.0, 2.5}},
		{Name: "timeStop", Values: []float64{5, 10}},
	}
}

// WithParams returns a fresh *bollSnapback starting from the defaults,
// overridden by any period/k/timeStop entries in params. It never mutates
// the receiver. Unknown keys, non-integral or <2 period, non-positive k,
// and non-integral or <1 timeStop are all rejected.
func (s *bollSnapback) WithParams(params map[string]float64) (strategy.Strategy, error) {
	next := newBollSnapback()

	for name, v := range params {
		switch name {
		case "period":
			if v != math.Trunc(v) {
				return nil, fmt.Errorf("boll-snapback: period must be an integral value, got %v", v)
			}
			next.period = int(v)
		case "k":
			next.k = v
		case "timeStop":
			if v != math.Trunc(v) {
				return nil, fmt.Errorf("boll-snapback: timeStop must be an integral value, got %v", v)
			}
			next.timeStop = int(v)
		default:
			return nil, fmt.Errorf("boll-snapback: unknown parameter %q", name)
		}
	}

	if next.period < 2 {
		return nil, fmt.Errorf("boll-snapback: period must be >= 2, got %d", next.period)
	}
	if next.k <= 0 {
		return nil, fmt.Errorf("boll-snapback: k must be positive, got %v", next.k)
	}
	if next.timeStop < 1 {
		return nil, fmt.Errorf("boll-snapback: timeStop must be >= 1, got %d", next.timeStop)
	}

	return next, nil
}

var _ strategy.Tunable = (*bollSnapback)(nil)

func init() {
	strategy.Register(func() strategy.Strategy { return newBollSnapback() })
}
