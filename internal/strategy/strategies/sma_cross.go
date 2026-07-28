// Package strategies holds concrete Strategy implementations, one file per
// strategy, each self-registering via init(). Importing this package for
// side effects (blank import) is required to populate the strategy
// registry.
package strategies

import (
	"fmt"
	"math"

	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// trials: counted mechanically per walk-forward run — grid is fast x slow
// (3x3 = 9 combos/fold), minus any combo with fast >= slow, which WithParams
// rejects and the harness excludes from the trial count.

// smaCross is a classic 50/200 SMA "golden cross" trend-following strategy.
// It goes long when the 50-day SMA crosses above the 200-day SMA and
// flattens when it crosses back below.
//
// Derive-don't-shadow: the strategy holds no mutable state at all. The
// crossing edge is derived from history (today's SMAs vs. yesterday's SMAs
// computed from closes[:len-1]), and position state is derived from the
// actual portfolio, so a rejected or unfilled order can never leave the
// strategy believing something the book doesn't.
type smaCross struct {
	fast, slow int
}

func newSMACross() *smaCross {
	return &smaCross{fast: 50, slow: 200}
}

func (s *smaCross) Name() string { return "sma-cross" }

func (s *smaCross) Description() string {
	return "50/200 SMA golden cross trend following: full long on bullish cross, flat on bearish cross."
}

func (s *smaCross) Horizon() strategy.Horizon { return strategy.Long }

func (s *smaCross) WarmupBars() int { return s.slow }

func (s *smaCross) OnBar(ctx *strategy.Context, bar domain.Bar) []domain.Signal {
	closes := strategy.Closes(ctx.History())

	fastNow, okFastNow := strategy.SMA(closes, s.fast)
	slowNow, okSlowNow := strategy.SMA(closes, s.slow)
	if !okFastNow || !okSlowNow {
		return nil
	}

	// Yesterday's SMAs, derived from history rather than cached: this is
	// what makes the crossing detection stateless. With WarmupBars == slow
	// the first OnBar call sees slow+1 bars, so closes[:len-1] always has
	// enough data here.
	prevCloses := closes[:len(closes)-1]
	fastPrev, okFastPrev := strategy.SMA(prevCloses, s.fast)
	slowPrev, okSlowPrev := strategy.SMA(prevCloses, s.slow)
	if !okFastPrev || !okSlowPrev {
		return nil
	}

	above := fastNow > slowNow
	prevAbove := fastPrev > slowPrev

	// Actual position, from the portfolio — never assumed from past signals.
	qty := int64(0)
	if pos, ok := ctx.Portfolio().Positions[bar.Symbol]; ok {
		qty = pos.Qty
	}

	if above && !prevAbove && qty == 0 {
		// Golden cross while flat: go long. (The risk manager owns sizing;
		// 1.0 expresses full conviction and will be clamped to its cap.)
		return []domain.Signal{{Symbol: bar.Symbol, TargetWeight: 1.0}}
	}
	if !above && prevAbove && qty > 0 {
		// Death cross while long: flatten.
		return []domain.Signal{{Symbol: bar.Symbol, TargetWeight: 0.0}}
	}

	return nil
}

// TargetWeights is the level-form (not edge-triggered) equivalent of OnBar's
// crossing logic, for use by the ensemble allocator (strategy.TargetWeighter
// seam): it answers "what does this strategy want RIGHT NOW", derived
// purely from history, with no dependence on any live position. OnBar
// remains unchanged for solo runs — it still only acts on the crossing edge
// (golden cross to enter, death cross to exit) — while TargetWeights reports
// the LEVEL: 1.0 whenever fast > slow at the last bar (regardless of
// whether a cross just happened), else 0. It must not read ctx.Portfolio().
func (s *smaCross) TargetWeights(ctx *strategy.Context) map[string]float64 {
	history := ctx.History()
	if len(history) == 0 {
		return map[string]float64{}
	}
	sym := history[len(history)-1].Symbol

	closes := strategy.Closes(history)
	fastNow, okFast := strategy.SMA(closes, s.fast)
	slowNow, okSlow := strategy.SMA(closes, s.slow)
	if !okFast || !okSlow {
		return map[string]float64{}
	}

	if fastNow > slowNow {
		return map[string]float64{sym: 1.0}
	}
	return map[string]float64{}
}

// ParamSpace declares the discrete grid walk-forward optimization searches
// over: 3 fast periods x 3 slow periods (9 combos, minus any with
// fast >= slow, which WithParams rejects).
func (s *smaCross) ParamSpace() []strategy.ParamDef {
	return []strategy.ParamDef{
		{Name: "fast", Values: []float64{20, 50, 100}},
		{Name: "slow", Values: []float64{100, 150, 200}},
	}
}

// WithParams returns a fresh *smaCross starting from the 50/200 defaults,
// overridden by any fast/slow entries in params. It never mutates the
// receiver. Unknown keys, non-integral values, non-positive periods, and
// fast >= slow are all rejected.
func (s *smaCross) WithParams(params map[string]float64) (strategy.Strategy, error) {
	next := newSMACross()

	for name, v := range params {
		switch name {
		case "fast":
			if v != math.Trunc(v) {
				return nil, fmt.Errorf("sma-cross: fast must be an integral value, got %v", v)
			}
			next.fast = int(v)
		case "slow":
			if v != math.Trunc(v) {
				return nil, fmt.Errorf("sma-cross: slow must be an integral value, got %v", v)
			}
			next.slow = int(v)
		default:
			return nil, fmt.Errorf("sma-cross: unknown parameter %q", name)
		}
	}

	if next.fast <= 0 {
		return nil, fmt.Errorf("sma-cross: fast must be positive, got %d", next.fast)
	}
	if next.slow <= 0 {
		return nil, fmt.Errorf("sma-cross: slow must be positive, got %d", next.slow)
	}
	if next.fast >= next.slow {
		return nil, fmt.Errorf("sma-cross: fast (%d) must be less than slow (%d)", next.fast, next.slow)
	}

	return next, nil
}

var _ strategy.Tunable = (*smaCross)(nil)
var _ strategy.TargetWeighter = (*smaCross)(nil)

func init() {
	strategy.Register(func() strategy.Strategy { return newSMACross() })
}
