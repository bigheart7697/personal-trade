// Package strategy defines the Strategy interface, a self-registration
// registry, shared indicators, and the concrete strategies under
// strategies/. Strategies express intent only (target portfolio weights);
// internal/risk is solely responsible for turning that intent into sized
// orders.
package strategy

import (
	"time"

	"tradeforge/internal/domain"
)

// Horizon classifies a strategy's holding-period family. The two families
// are kept visually and organizationally separate throughout the platform
// (registry listing, CLI --list, dashboard sections).
type Horizon int

const (
	// Short horizon: strategies that typically hold for days.
	Short Horizon = iota
	// Long horizon: strategies that typically hold for weeks to months.
	Long
)

// String implements fmt.Stringer.
func (h Horizon) String() string {
	switch h {
	case Short:
		return "short-term"
	case Long:
		return "long-term"
	default:
		return "unknown"
	}
}

// Context is the read-only view a Strategy receives on each bar: completed
// history (oldest first, ending at and including the current bar) and the
// current portfolio state. Strategies must never be given data beyond the
// current bar — that is the no-lookahead contract enforced by the backtest
// engine.
//
// A Context is either single-symbol (built via NewContext, used by the
// engine's Bars path) or multi-symbol (built via NewMultiContext, used by
// the engine's BarSets/runMulti path). Only one of history / histories is
// ever populated; History and HistoryOf are the respective accessors.
type Context struct {
	history     []domain.Bar
	histories   map[string][]domain.Bar
	portfolio   *domain.Portfolio
	positionAge map[string]int
}

// NewContext builds a single-symbol Context. history must already be
// truncated to end at the current bar (inclusive) by the caller (the
// backtest engine).
func NewContext(history []domain.Bar, portfolio *domain.Portfolio) *Context {
	return &Context{history: history, portfolio: portfolio}
}

// NewMultiContext builds a multi-symbol Context for a MultiSymbol strategy's
// OnUniverseBar call. histories maps each universe symbol to its completed
// bars up to and including the current master-clock tick (already truncated
// by the caller — the backtest engine's runMulti). Use HistoryOf to read a
// symbol's history from a multi Context; History always returns nil here.
func NewMultiContext(histories map[string][]domain.Bar, portfolio *domain.Portfolio) *Context {
	return &Context{histories: histories, portfolio: portfolio}
}

// History returns all completed bars up to and including the current bar,
// oldest first, for a single-symbol Context. Callers must not mutate the
// returned slice. On a multi-symbol Context (built via NewMultiContext) it
// always returns nil — multi-symbol strategies must use HistoryOf instead.
func (c *Context) History() []domain.Bar {
	return c.history
}

// HistoryOf returns symbol's completed bars up to and including the current
// tick, oldest first, or nil if symbol is unknown to this Context. Callers
// must not mutate the returned slice.
//
// On a multi-symbol Context (built via NewMultiContext) this looks up
// symbol in the histories map. On a single-symbol Context (built via
// NewContext) it returns History() when symbol matches the symbol of the
// context's own history (checked via the last bar, i.e. the current bar),
// and nil otherwise — this lets code written against HistoryOf work
// uniformly regardless of which kind of Context it was given.
func (c *Context) HistoryOf(symbol string) []domain.Bar {
	if c.histories != nil {
		return c.histories[symbol]
	}
	if len(c.history) == 0 {
		return nil
	}
	if c.history[len(c.history)-1].Symbol != symbol {
		return nil
	}
	return c.history
}

// Portfolio returns the current portfolio state. It is exposed for
// strategies that condition on existing positions (e.g. rsi2's time-based
// exit); strategies must not mutate it directly — only risk.Manager and the
// engine change portfolio state.
func (c *Context) Portfolio() *domain.Portfolio {
	return c.portfolio
}

// WithPositionAge attaches per-symbol position ages (see PositionAge) to the
// Context and returns it, so environments can chain it onto NewContext /
// NewMultiContext without disturbing existing call sites. A Context built
// without WithPositionAge (or with a nil map) simply reports age 0 for every
// symbol — "unknown", the safe default.
func (c *Context) WithPositionAge(ages map[string]int) *Context {
	c.positionAge = ages
	return c
}

// PositionAge returns the number of completed bars for which the symbol's
// position has been continuously open, COUNTING THE CURRENT BAR: on the
// first OnBar call after an entry fill it returns 1, matching the historic
// barsHeld convention (increment-then-check). Returns 0 when flat or when
// the age is unknown. The value is SUPPLIED by the environment (backtest
// engine from its fill log; paper session from the persisted order ledger)
// — strategies must use this instead of shadowing their own counters,
// which silently reset across process restarts (found live 2026-07-07:
// time-stops could never fire in daily paper sessions).
//
// Degradation contract: 0 while a position is demonstrably held means "age
// unknown" — time-stop logic must treat the position as just-entered (guard
// with age > 0) rather than force an exit, so a missing ledger never
// liquidates a healthy position.
func (c *Context) PositionAge(symbol string) int {
	if c.positionAge == nil {
		return 0
	}
	return c.positionAge[symbol]
}

// Strategy is the interface every trading strategy implements. OnBar is
// called once per completed bar (after warmup) and returns zero or more
// Signals expressing desired target portfolio weights; it must not read any
// data beyond the current bar.
type Strategy interface {
	Name() string
	Description() string
	Horizon() Horizon
	WarmupBars() int
	OnBar(ctx *Context, bar domain.Bar) []domain.Signal
}

// MultiSymbol is implemented by strategies that trade a fixed universe of
// symbols cross-sectionally (e.g. dual momentum, pairs trading). Universe()
// declares the symbol list the engine must feed — fixed for the life of a
// run. OnUniverseBar is called once per master-clock tick with the bars that
// completed at that tick (bars contains only symbols that actually have a
// bar that day; a symbol can be absent on any given tick). Signals may
// reference any universe symbol, not just ones present in bars that tick.
//
// For a MultiSymbol strategy, the engine never calls OnBar — only
// OnUniverseBar. Implementers must still satisfy the Strategy interface;
// the documented convention is for OnBar to be a stub that returns nil.
// Ordinary single-symbol strategies ignore this interface entirely.
type MultiSymbol interface {
	Strategy
	Universe() []string
	OnUniverseBar(ctx *Context, date time.Time, bars map[string]domain.Bar) []domain.Signal
}

// TargetWeighter is an OPTIONAL interface for strategies whose desired
// portfolio state is a pure function of price history — no dependence on
// their own live position state. TargetWeights returns the strategy's
// desired weight per symbol (long-only, [0,1], missing = 0) given only
// completed history. Implementations must be deterministic and must not
// read ctx.Portfolio() — the ensemble allocator calls this with virtual
// books that are none of the member's business.
type TargetWeighter interface {
	Strategy
	TargetWeights(ctx *Context) map[string]float64
}
