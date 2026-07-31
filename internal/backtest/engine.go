// Package backtest implements the event-driven simulation loop: bars flow
// into a strategy, its signals flow into the risk manager for sizing and
// approval, approved orders fill at the next bar's open via a broker, and
// fills update the portfolio. This is the same data flow used by paper and
// live modes (only the BarSource and Broker implementations change), which
// is what keeps backtest and live behavior from diverging.
package backtest

import (
	"fmt"
	"sort"
	"time"

	"tradeforge/internal/broker"
	"tradeforge/internal/domain"
	"tradeforge/internal/metrics"
	"tradeforge/internal/risk"
	"tradeforge/internal/strategy"
)

// Config configures a single backtest run. Exactly one of Bars / BarSets
// must be set: Bars drives the original single-symbol loop (Run's body
// below), BarSets drives the multi-symbol master-clock loop (runMulti) and
// requires Strategy to implement strategy.MultiSymbol.
type Config struct {
	Strategy    strategy.Strategy
	Bars        []domain.Bar
	BarSets     map[string][]domain.Bar
	InitialCash float64

	// SimBroker settings.
	SlippageBps        float64
	CommissionPerShare float64
	MinCommission      float64
}

// maxOrderAge is the number of master-clock ticks (runMulti only) a queued
// order may wait for its symbol to trade before it expires and is logged as
// a Rejection. An order's age starts at 0 when queued; it is incremented
// once per tick its symbol has no bar. This is the multi-symbol
// generalization of the single-symbol path's "fill at the very next bar"
// rule — a symbol that trades every tick always fills at age 0 (its next
// bar), matching Run's behavior exactly.
const maxOrderAge = 5

// Rejection records a signal the risk manager declined to convert into an
// order, with the reason why.
type Rejection struct {
	Time   time.Time
	Symbol string
	Reason string
}

// Clamp records a signal whose target weight the risk manager altered
// before sizing (capped at the max position weight, or zeroed because
// shorts are not supported). Clamping is not a rejection — the order may
// still have been approved at the smaller size — but it must be visible so
// a strategy claiming 50% while trading 20% is never silent.
type Clamp struct {
	Time   time.Time
	Symbol string
	Reason string
}

// Result is everything a backtest run produces.
type Result struct {
	StrategyName string
	Horizon      strategy.Horizon
	PeriodStart  time.Time
	PeriodEnd    time.Time

	EquityCurve []metrics.EquityPoint
	Exposure    []bool // same length/indexing as EquityCurve; Exposure[i] is whether the portfolio held a non-flat position at EquityCurve[i]
	Fills       []domain.Fill
	Rejections  []Rejection
	Clamps      []Clamp
	Metrics     metrics.Metrics
}

// Run executes cfg's strategy over cfg.Bars, producing a daily equity
// curve, fill log, rejection log, clamp log, and computed metrics.
//
// Causality per bar i, in this exact order:
//
//  1. Execute orders queued from bar i-1 — they fill at cfg.Bars[i].Open
//     (plus slippage/commission) and the fills are applied to the
//     portfolio.
//  2. Mark equity at cfg.Bars[i].Close. The equity point dated bar i
//     therefore reflects only trades executed at or before bar i's open —
//     never a trade that happens the next day.
//  3. If i >= warmup and i is not the last bar, run the strategy on bar i
//     (it sees cfg.Bars[:i+1], oldest first — the no-lookahead contract),
//     pass each signal through risk.ApproveOrder using bar i's close as
//     the sizing reference price, and QUEUE approved orders for execution
//     at bar i+1's open in the next iteration.
func Run(cfg Config) (Result, error) {
	if cfg.Strategy == nil {
		return Result{}, fmt.Errorf("backtest: nil strategy")
	}
	if cfg.InitialCash <= 0 {
		return Result{}, fmt.Errorf("backtest: initial cash must be positive, got %.2f", cfg.InitialCash)
	}
	if len(cfg.Bars) > 0 && len(cfg.BarSets) > 0 {
		return Result{}, fmt.Errorf("backtest: exactly one of Bars / BarSets may be set, got both")
	}

	if cfg.BarSets != nil {
		return runMulti(cfg)
	}

	if len(cfg.Bars) == 0 {
		return Result{}, fmt.Errorf("backtest: no bars provided")
	}
	if _, ok := cfg.Strategy.(strategy.MultiSymbol); ok {
		return Result{}, fmt.Errorf("backtest: strategy %q is multi-symbol; use --universe (BarSets), not --data (Bars)", cfg.Strategy.Name())
	}

	warmup := cfg.Strategy.WarmupBars()
	if warmup < 0 {
		warmup = 0
	}

	port := domain.NewPortfolio(cfg.InitialCash)
	riskMgr := risk.NewManager()
	simBroker := broker.NewSimBroker(cfg.SlippageBps)
	if cfg.CommissionPerShare > 0 {
		simBroker.CommissionPerShare = cfg.CommissionPerShare
	}
	if cfg.MinCommission > 0 {
		simBroker.MinCommission = cfg.MinCommission
	}

	var (
		equityCurve []metrics.EquityPoint
		fills       []domain.Fill
		rejections  []Rejection
		clamps      []Clamp
		exposed     []bool
		pending     []domain.Order // orders queued at bar i-1, to fill at bar i's open
		equityPeak  = cfg.InitialCash
		orderSeq    int

		// entryBarIdx is the bar index at which the current position was
		// opened (a fill took the symbol's qty from 0 to nonzero), or -1
		// while flat. It feeds Context.PositionAge — the DERIVED position
		// age strategies use for time-stops instead of shadowing their own
		// counters (derive-don't-shadow).
		entryBarIdx = -1
	)

	n := len(cfg.Bars)

	for i := 0; i < n; i++ {
		bar := cfg.Bars[i]

		// (1) Execute orders queued from the previous bar at THIS bar's
		// open, and apply the fills before anything else happens today.
		for _, order := range pending {
			simBroker.AvailableCash = port.Cash
			fill, err := simBroker.SubmitAtOpen(order, bar)
			if err != nil {
				rejections = append(rejections, Rejection{
					Time:   bar.Time,
					Symbol: order.Symbol,
					Reason: err.Error(),
				})
				continue
			}
			qtyBefore := positionQty(port, order.Symbol)
			port.ApplyFill(fill)
			qtyAfter := positionQty(port, order.Symbol)
			switch {
			case qtyBefore == 0 && qtyAfter != 0:
				entryBarIdx = i // position opened at this bar's open
			case qtyBefore != 0 && qtyAfter == 0:
				entryBarIdx = -1 // position closed; back to flat
			}
			fills = append(fills, fill)
		}
		pending = pending[:0]

		// (2) Mark equity at this bar's close. Fills from step (1) are
		// included; orders queued in step (3) below are NOT — they haven't
		// happened yet.
		eq := port.Equity(map[string]float64{bar.Symbol: bar.Close})
		if eq > equityPeak {
			equityPeak = eq
		}
		equityCurve = append(equityCurve, metrics.EquityPoint{Time: bar.Time, Equity: eq})
		exposed = append(exposed, isExposed(port))

		// (3) Strategy sees completed bar i and queues orders for bar i+1.
		// Skipped during warmup and on the last bar (no next open to fill
		// at).
		if i >= warmup && i < n-1 {
			history := cfg.Bars[:i+1] // no-lookahead: bars up to and including i only

			// Derived position age, counting the current bar: an entry fill
			// applies at bar e's open (step 1), so the first OnBar with the
			// position — this same bar e — sees age e-e+1 = 1, matching the
			// historic increment-then-check barsHeld convention. 0 = flat.
			age := 0
			if entryBarIdx >= 0 {
				age = i - entryBarIdx + 1
			}
			ctx := strategy.NewContext(history, port).WithPositionAge(map[string]int{bar.Symbol: age})
			signals := cfg.Strategy.OnBar(ctx, bar)

			for _, sig := range signals {
				order, approved, reason, clampReason := riskMgr.ApproveOrder(sig, port, bar.Close, equityPeak)
				if clampReason != "" {
					clamps = append(clamps, Clamp{
						Time:   bar.Time,
						Symbol: sig.Symbol,
						Reason: clampReason,
					})
				}
				if !approved {
					rejections = append(rejections, Rejection{
						Time:   bar.Time,
						Symbol: sig.Symbol,
						Reason: reason,
					})
					continue
				}

				orderSeq++
				order.ID = fmt.Sprintf("%s-%d", cfg.Strategy.Name(), orderSeq)
				order.Time = bar.Time
				pending = append(pending, order)
			}
		}
	}

	result := Result{
		StrategyName: cfg.Strategy.Name(),
		Horizon:      cfg.Strategy.Horizon(),
		EquityCurve:  equityCurve,
		Exposure:     exposed,
		Fills:        fills,
		Rejections:   rejections,
		Clamps:       clamps,
	}
	result.PeriodStart = cfg.Bars[0].Time
	result.PeriodEnd = cfg.Bars[n-1].Time
	result.Metrics = metrics.Compute(equityCurve, len(fills), exposed)

	return result, nil
}

func isExposed(port *domain.Portfolio) bool {
	for _, pos := range port.Positions {
		if pos.Qty != 0 {
			return true
		}
	}
	return false
}

// positionQty returns port's current quantity for symbol (0 if the symbol has
// no position entry). Used around ApplyFill to detect the flat->open and
// open->flat transitions that drive the derived position age.
func positionQty(port *domain.Portfolio, symbol string) int64 {
	if pos, ok := port.Positions[symbol]; ok {
		return pos.Qty
	}
	return 0
}

// pendingOrder is a queued order awaiting a fill on its own symbol's next
// bar, tracked with an age in master-clock ticks so it can expire if its
// symbol goes quiet (see maxOrderAge).
type pendingOrder struct {
	order domain.Order
	age   int
}

// runMulti executes cfg.Strategy (which must implement strategy.MultiSymbol)
// over cfg.BarSets on a master clock: the sorted union of every bar time
// across all universe symbols. See docs/ARCHITECTURE.md "Multi-symbol
// engine" for the full causality design; this implementation follows it
// tick by tick:
//
//  1. FILLS: for each universe symbol (sorted order) with a bar at this
//     tick, fill its pending orders (FIFO) at that bar's open via the
//     SimBroker and apply the fills. Symbols with no bar this tick instead
//     have their pending orders aged by one tick; an order that reaches
//     maxOrderAge without a fill expires (logged as a Rejection).
//  2. MARK: update each present symbol's latest known close, then mark
//     portfolio equity using every symbol's latest known close so far
//     (stale closes are used for symbols absent this tick).
//  3. SIGNALS: once warmup has elapsed and this is not the last tick, build
//     each universe symbol's history up to and including this tick (via a
//     per-symbol cursor, no re-scanning), call OnUniverseBar, sort the
//     returned signals by symbol (the determinism rule — map/strategy
//     iteration order must never reach results), and run each through
//     risk.Manager.ApproveOrder exactly like the single-symbol path,
//     queuing approved orders onto their own symbol's pending queue.
func runMulti(cfg Config) (Result, error) {
	ms, ok := cfg.Strategy.(strategy.MultiSymbol)
	if !ok {
		return Result{}, fmt.Errorf("backtest: BarSets requires a strategy.MultiSymbol strategy, got %q which does not implement it", cfg.Strategy.Name())
	}

	universe := ms.Universe()
	if len(universe) == 0 {
		return Result{}, fmt.Errorf("backtest: strategy %q returned an empty Universe()", cfg.Strategy.Name())
	}

	var missing []string
	for _, sym := range universe {
		series, ok := cfg.BarSets[sym]
		if !ok {
			missing = append(missing, sym)
			continue
		}
		if len(series) == 0 {
			return Result{}, fmt.Errorf("backtest: BarSets[%q] is empty", sym)
		}
	}
	if len(missing) > 0 {
		return Result{}, fmt.Errorf("backtest: BarSets is missing universe symbol(s): %v", missing)
	}

	// Build the master clock: the sorted union of every bar time across all
	// universe symbols. Only the universe's own symbols are considered here
	// (cfg.BarSets may carry extra keys the strategy doesn't use), matching
	// this function's pre-M3 behavior exactly.
	clock := MasterClock(subsetBarSets(cfg.BarSets, universe))

	// barAt[sym] maps tick time -> bar index in cfg.BarSets[sym], so each
	// tick can look up whether/where a symbol has a bar in O(1) without
	// re-scanning.
	barAt := make(map[string]map[time.Time]int, len(universe))
	for _, sym := range universe {
		idx := make(map[time.Time]int, len(cfg.BarSets[sym]))
		for i, b := range cfg.BarSets[sym] {
			idx[b.Time] = i
		}
		barAt[sym] = idx
	}

	// Warmup counts MASTER-CLOCK ticks, not per-symbol bars: a symbol that
	// listed late may have fewer than WarmupBars() bars of its own history
	// when OnUniverseBar first fires. MultiSymbol strategies must therefore
	// guard per-symbol history length themselves (dual-momentum's
	// len(hist) < lookback+1 eligibility check is the pattern).
	warmup := cfg.Strategy.WarmupBars()
	if warmup < 0 {
		warmup = 0
	}

	port := domain.NewPortfolio(cfg.InitialCash)
	riskMgr := risk.NewManager()
	simBroker := broker.NewSimBroker(cfg.SlippageBps)
	if cfg.CommissionPerShare > 0 {
		simBroker.CommissionPerShare = cfg.CommissionPerShare
	}
	if cfg.MinCommission > 0 {
		simBroker.MinCommission = cfg.MinCommission
	}

	var (
		equityCurve []metrics.EquityPoint
		fills       []domain.Fill
		rejections  []Rejection
		clamps      []Clamp
		exposed     []bool
		equityPeak  = cfg.InitialCash
		orderSeq    int
	)

	pending := make(map[string][]pendingOrder, len(universe)) // symbol -> queued orders
	cursor := make(map[string]int, len(universe))             // symbol -> next unconsumed bar index
	latestClose := make(map[string]float64, len(universe))

	// entryTick maps a held symbol to the master-clock tick index at which
	// its position was opened (fill took qty from 0 to nonzero); the entry is
	// deleted when the position returns to flat. It feeds Context.PositionAge
	// — ages here count master-clock TICKS held (the multi-symbol analog of
	// bars), counting the current tick.
	entryTick := make(map[string]int, len(universe))

	n := len(clock)

	for tickIdx, t := range clock {
		// (1) FILLS.
		for _, sym := range universe {
			idx, hasBar := barAt[sym][t]
			if !hasBar {
				// No bar for this symbol today: age its pending orders and
				// expire any that have waited too long.
				queue := pending[sym]
				kept := queue[:0]
				for _, po := range queue {
					po.age++
					if po.age >= maxOrderAge {
						rejections = append(rejections, Rejection{
							Time:   t,
							Symbol: sym,
							Reason: fmt.Sprintf("expired: no bar for %s within %d ticks", sym, maxOrderAge),
						})
						continue
					}
					kept = append(kept, po)
				}
				pending[sym] = kept
				continue
			}

			bar := cfg.BarSets[sym][idx]
			queue := pending[sym]
			pending[sym] = nil
			for _, po := range queue {
				simBroker.AvailableCash = port.Cash
				fill, err := simBroker.SubmitAtOpen(po.order, bar)
				if err != nil {
					rejections = append(rejections, Rejection{
						Time:   bar.Time,
						Symbol: po.order.Symbol,
						Reason: err.Error(),
					})
					continue
				}
				qtyBefore := positionQty(port, po.order.Symbol)
				port.ApplyFill(fill)
				qtyAfter := positionQty(port, po.order.Symbol)
				switch {
				case qtyBefore == 0 && qtyAfter != 0:
					entryTick[po.order.Symbol] = tickIdx // position opened at this tick's open
				case qtyBefore != 0 && qtyAfter == 0:
					delete(entryTick, po.order.Symbol) // position closed; back to flat
				}
				fills = append(fills, fill)
			}
		}

		// (2) MARK.
		for _, sym := range universe {
			if idx, hasBar := barAt[sym][t]; hasBar {
				latestClose[sym] = cfg.BarSets[sym][idx].Close
				cursor[sym] = idx + 1 // advance past the bar that just completed
			}
		}
		eq := port.Equity(copyPriceMap(latestClose))
		if eq > equityPeak {
			equityPeak = eq
		}
		equityCurve = append(equityCurve, metrics.EquityPoint{Time: t, Equity: eq})
		exposed = append(exposed, isExposed(port))

		// (3) SIGNALS.
		if tickIdx >= warmup && tickIdx < n-1 {
			barsAtT := make(map[string]domain.Bar)
			histories := make(map[string][]domain.Bar, len(universe))
			for _, sym := range universe {
				if idx, hasBar := barAt[sym][t]; hasBar {
					barsAtT[sym] = cfg.BarSets[sym][idx]
				}
				histories[sym] = cfg.BarSets[sym][:cursor[sym]]
			}

			// Derived position ages, counting the current tick (see the
			// single-symbol path's step 3 for the age = 1 convention).
			ages := make(map[string]int, len(entryTick))
			for sym, e := range entryTick {
				ages[sym] = tickIdx - e + 1
			}

			ctx := strategy.NewMultiContext(histories, port).WithPositionAge(ages)
			signals := ms.OnUniverseBar(ctx, t, barsAtT)

			sort.SliceStable(signals, func(i, j int) bool { return signals[i].Symbol < signals[j].Symbol })

			for _, sig := range signals {
				refPrice, known := latestClose[sig.Symbol]
				if !known {
					rejections = append(rejections, Rejection{
						Time:   t,
						Symbol: sig.Symbol,
						Reason: "no price history yet",
					})
					continue
				}

				order, approved, reason, clampReason := riskMgr.ApproveOrder(sig, port, refPrice, equityPeak)
				if clampReason != "" {
					clamps = append(clamps, Clamp{
						Time:   t,
						Symbol: sig.Symbol,
						Reason: clampReason,
					})
				}
				if !approved {
					rejections = append(rejections, Rejection{
						Time:   t,
						Symbol: sig.Symbol,
						Reason: reason,
					})
					continue
				}

				orderSeq++
				order.ID = fmt.Sprintf("%s-%d", cfg.Strategy.Name(), orderSeq)
				order.Time = t
				pending[sig.Symbol] = append(pending[sig.Symbol], pendingOrder{order: order})
			}
		}
	}

	result := Result{
		StrategyName: cfg.Strategy.Name(),
		Horizon:      cfg.Strategy.Horizon(),
		EquityCurve:  equityCurve,
		Exposure:     exposed,
		Fills:        fills,
		Rejections:   rejections,
		Clamps:       clamps,
	}
	result.PeriodStart = clock[0]
	result.PeriodEnd = clock[n-1]
	result.Metrics = metrics.Compute(equityCurve, len(fills), exposed)

	return result, nil
}

// copyPriceMap returns a shallow copy of m so callers that stash a
// reference (Portfolio.Equity does not, but the contract is easier to keep
// if every call site gets its own map) never observe later mutations.
func copyPriceMap(m map[string]float64) map[string]float64 {
	out := make(map[string]float64, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

// subsetBarSets returns a new map containing only barSets' entries for the
// given symbols, so MasterClock (and any other caller that must consider
// only a specific symbol set) never picks up unrelated keys a caller's
// BarSets map happens to also carry.
func subsetBarSets(barSets map[string][]domain.Bar, symbols []string) map[string][]domain.Bar {
	out := make(map[string][]domain.Bar, len(symbols))
	for _, sym := range symbols {
		out[sym] = barSets[sym]
	}
	return out
}

// MasterClock returns the sorted union of every bar time across all symbols
// in barSets: the tick sequence runMulti iterates. Exported so
// internal/eval can build the same clock when evaluating a MultiSymbol
// strategy (fold arithmetic there counts these clock ticks, not per-symbol
// bars — see docs/ARCHITECTURE.md "Multi-symbol engine").
func MasterClock(barSets map[string][]domain.Bar) []time.Time {
	tickSet := make(map[time.Time]struct{})
	for _, series := range barSets {
		for _, b := range series {
			tickSet[b.Time] = struct{}{}
		}
	}
	clock := make([]time.Time, 0, len(tickSet))
	for t := range tickSet {
		clock = append(clock, t)
	}
	sort.Slice(clock, func(i, j int) bool { return clock[i].Before(clock[j]) })
	return clock
}

// WindowBarSets returns, for each symbol in barSets, the contiguous subslice
// of bars with from <= Time <= to (both endpoints inclusive). Bars within
// each symbol's series must already be sorted by Time (every ingestion path
// guarantees this) so the window is found with a binary search rather than a
// linear scan. The returned slices share the original series' backing
// arrays — no copying.
//
// A symbol with zero bars in [from, to] is OMITTED from the result entirely
// (not included with an empty slice), because backtest.Config's BarSets
// validation rejects an empty series for a universe symbol. This means
// callers — eval's fold-windowing above all — must ensure every window is
// wide enough that each universe symbol has at least one bar in it; if a
// universe symbol has no bars in a window, the subsequent backtest.Run call
// will error on that missing/empty series, which is the correct, loud
// failure rather than a silently degraded run.
func WindowBarSets(barSets map[string][]domain.Bar, from, to time.Time) map[string][]domain.Bar {
	out := make(map[string][]domain.Bar, len(barSets))
	for sym, series := range barSets {
		start := sort.Search(len(series), func(i int) bool {
			return !series[i].Time.Before(from)
		})
		end := sort.Search(len(series), func(i int) bool {
			return series[i].Time.After(to)
		})
		if start >= end {
			continue
		}
		out[sym] = series[start:end]
	}
	return out
}
