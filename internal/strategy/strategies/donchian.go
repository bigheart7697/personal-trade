package strategies

import (
	"fmt"
	"math"

	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// trials: counted mechanically per walk-forward run — grid is entry x exit
// (3x3 = 9 combos/fold); entry=40/55/70 are all > exit=10/20/30, so
// WithParams's entry>exit constraint accepts all 9 combos (none excluded).

// donchian is a classic Donchian channel breakout: buy when price makes a
// new N-bar high, sell when it makes a new M-bar low ("turtle"-style trend
// following, long-only here).
//
// Derive-don't-shadow: the strategy holds no mutable state at all. Position
// state is derived from the actual portfolio (ctx.Portfolio()), and the
// channel is recomputed fresh from history on every call, so a rejected or
// unfilled order can never leave the strategy believing something the book
// doesn't.
//
// No-lookahead: both channels are built from the PREVIOUS `entry`/`exit`
// bars — history[:len-1] — deliberately excluding the current bar. Including
// today's own high/low in its own breakout channel would make the breakout
// trivially self-referential (today's high always >= itself).
type donchian struct {
	entry, exit int
}

func newDonchian() *donchian {
	return &donchian{entry: 55, exit: 20}
}

func (s *donchian) Name() string { return "donchian" }

func (s *donchian) Description() string {
	return "Donchian channel breakout: long on a new 55-bar high, exit on a new 20-bar low."
}

func (s *donchian) Horizon() strategy.Horizon { return strategy.Long }

// WarmupBars guarantees history[:len-1] has at least `entry` (resp. `exit`)
// bars on the very first OnBar call: the engine first calls OnBar with
// WarmupBars+1 bars of history, so excluding the current bar leaves
// WarmupBars = max(entry, exit)+1 prior bars — at least max(entry, exit),
// enough for whichever channel is wider.
func (s *donchian) WarmupBars() int {
	if s.entry >= s.exit {
		return s.entry + 1
	}
	return s.exit + 1
}

func (s *donchian) OnBar(ctx *strategy.Context, bar domain.Bar) []domain.Signal {
	history := ctx.History()

	// Actual position, from the portfolio — never assumed from past signals.
	qty := int64(0)
	if pos, ok := ctx.Portfolio().Positions[bar.Symbol]; ok {
		qty = pos.Qty
	}
	inPosition := qty > 0

	// Exclude the current bar: the channel is built strictly from history
	// prior to today, so today's own high/low can never create today's
	// breakout.
	prior := history[:len(history)-1]

	if inPosition {
		lows := strategy.Lows(prior)
		lowestLow, ok := strategy.MinN(lows, s.exit)
		if !ok {
			return nil
		}
		if bar.Close < lowestLow {
			return []domain.Signal{{Symbol: bar.Symbol, TargetWeight: 0.0}}
		}
		return nil
	}

	highs := strategy.Highs(prior)
	highestHigh, ok := strategy.MaxN(highs, s.entry)
	if !ok {
		return nil
	}
	if bar.Close > highestHigh {
		// Breakout while flat: go long. (The risk manager owns sizing; 1.0
		// expresses full conviction and will be clamped to its cap.)
		return []domain.Signal{{Symbol: bar.Symbol, TargetWeight: 1.0}}
	}

	return nil
}

// TargetWeights is the level-form equivalent of OnBar's breakout state
// machine, for use by the ensemble allocator (strategy.TargetWeighter seam).
// Because donchian's position is path-dependent (whether we're "in" depends
// on the last breakout/breakdown, not just the current bar), TargetWeights
// must replay the ENTIRE state machine deterministically from a flat start:
// walk forward from the first bar for which both channels are computable,
// entering on a close above the prior entry-channel high and exiting on a
// close below the prior exit-channel low, exactly mirroring OnBar's logic
// but driven by replayed state instead of ctx.Portfolio(). It must not read
// ctx.Portfolio() — this is the ensemble's own hypothetical, independent of
// any real position donchian(the same instance) might also be holding in a
// concurrent solo run.
//
// Cost note: this recomputes a MaxN/MinN channel for every bar in history on
// every call — O(n * window) per call. At daily-bar scale (thousands of
// bars) this is fast enough for a once-per-rebalance ensemble tick; it is
// deliberately not optimized to an incremental rolling channel.
func (s *donchian) TargetWeights(ctx *strategy.Context) map[string]float64 {
	history := ctx.History()
	if len(history) == 0 {
		return map[string]float64{}
	}
	sym := history[len(history)-1].Symbol

	warmup := s.WarmupBars()
	if len(history) < warmup {
		return map[string]float64{}
	}

	inPosition := false
	for i := warmup - 1; i < len(history); i++ {
		prior := history[:i] // strictly before bar i, same exclude-current-bar rule as OnBar
		bar := history[i]

		if inPosition {
			lows := strategy.Lows(prior)
			lowestLow, ok := strategy.MinN(lows, s.exit)
			if ok && bar.Close < lowestLow {
				inPosition = false
			}
			continue
		}

		highs := strategy.Highs(prior)
		highestHigh, ok := strategy.MaxN(highs, s.entry)
		if ok && bar.Close > highestHigh {
			inPosition = true
		}
	}

	if inPosition {
		return map[string]float64{sym: 1.0}
	}
	return map[string]float64{}
}

// ParamSpace declares the discrete grid walk-forward optimization searches
// over: 3 entry windows x 3 exit windows (9 combos, minus any with
// entry <= exit, which WithParams rejects).
func (s *donchian) ParamSpace() []strategy.ParamDef {
	return []strategy.ParamDef{
		{Name: "entry", Values: []float64{40, 55, 70}},
		{Name: "exit", Values: []float64{10, 20, 30}},
	}
}

// WithParams returns a fresh *donchian starting from the 55/20 defaults,
// overridden by any entry/exit entries in params. It never mutates the
// receiver. Unknown keys, non-integral values, entry < 2, exit < 2, and
// entry <= exit are all rejected.
func (s *donchian) WithParams(params map[string]float64) (strategy.Strategy, error) {
	next := newDonchian()

	for name, v := range params {
		switch name {
		case "entry":
			if v != math.Trunc(v) {
				return nil, fmt.Errorf("donchian: entry must be an integral value, got %v", v)
			}
			next.entry = int(v)
		case "exit":
			if v != math.Trunc(v) {
				return nil, fmt.Errorf("donchian: exit must be an integral value, got %v", v)
			}
			next.exit = int(v)
		default:
			return nil, fmt.Errorf("donchian: unknown parameter %q", name)
		}
	}

	if next.entry < 2 {
		return nil, fmt.Errorf("donchian: entry must be >= 2, got %d", next.entry)
	}
	if next.exit < 2 {
		return nil, fmt.Errorf("donchian: exit must be >= 2, got %d", next.exit)
	}
	if next.entry <= next.exit {
		return nil, fmt.Errorf("donchian: entry (%d) must be greater than exit (%d)", next.entry, next.exit)
	}

	return next, nil
}

var _ strategy.Tunable = (*donchian)(nil)
var _ strategy.TargetWeighter = (*donchian)(nil)

func init() {
	strategy.Register(func() strategy.Strategy { return newDonchian() })
}
