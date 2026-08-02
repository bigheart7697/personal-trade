// Shared single-series/multi-series resolution for WalkForward and KFold.
// Both evaluation methods accept either a single bar series (Bars) or a
// multi-symbol bar-set map (BarSets), and both need the exact same
// resolution rules: exactly one of the two set, a validated benchmark for
// the multi path, and a uniform way to turn a "bar index window" into a
// backtest.Config and a set of boundary times regardless of which mode is
// active. Putting that logic in one place is what lets the fold-arithmetic
// (walk-forward's sliding window, k-fold's span/purge/embargo) stay written
// once in terms of abstract indices [a,b) and work unmodified against master
// clock ticks instead of single-series bar indices.
//
// Single-series behavior is untouched byte-for-byte: resolvedSeries's Bars
// path slices cfg.Bars directly and produces the exact same
// backtest.Config{Bars: ...} the pre-M3 code built inline.
package eval

import (
	"fmt"
	"time"

	"tradeforge/internal/backtest"
	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// resolvedSeries is the common view WalkForward and KFold fold-arithmetic
// operates on: n abstract "ticks" (single-series bar indices, or
// master-clock indices for a multi-symbol run), a way to build the
// backtest.Config for the window [a,b) of those ticks, the boundary time for
// any tick (for FoldResult/KFoldFold's reported dates), and the benchmark
// bars used for the buy-and-hold chain / regime breakdown / random baseline.
type resolvedSeries struct {
	n int

	// multi is true for a BarSets-driven run. Fold-arithmetic behaves
	// identically either way (it only ever consumes n and the two methods
	// below); multi is exposed for error-message wording and because the
	// benchmark's OOS-coverage check only applies to the multi path (a
	// single-series benchmark IS the traded series, so it trivially covers
	// every OOS tick).
	multi bool

	// timeAt returns the boundary time for tick index i (0 <= i < n).
	timeAt func(i int) time.Time

	// windowConfig returns the backtest.Config fields (Bars XOR BarSets)
	// for the tick window [a, b).
	windowConfig func(a, b int) backtest.Config

	// benchSymbol is the resolved benchmark symbol label ("" for
	// single-series, where the traded series itself is the benchmark).
	benchSymbol string

	// benchBarsForWindow returns the benchmark's own bars restricted to the
	// tick window [a, b) — used to build each fold's buy-and-hold segment.
	// For single-series this is identical to windowConfig's Bars.
	benchBarsForWindow func(a, b int) []domain.Bar

	// fullBenchBars returns the benchmark's full bar series (single-series:
	// cfg.Bars; multi: the benchmark symbol's own BarSets entry), used for
	// RegimeBreakdown's underlying-bar classification.
	fullBenchBars func() []domain.Bar
}

// resolveSeries validates that exactly one of bars/barSets is set (mirroring
// the field docs on Config/KFoldConfig), resolves the benchmark symbol for a
// multi-symbol run, and returns a resolvedSeries ready for fold-arithmetic.
// factory is used only to pre-validate that a MultiSymbol strategy backs a
// BarSets config (a clearer error than letting the engine's own check
// surface on the first fold's train run).
func resolveSeries(factory func() strategy.Strategy, bars []domain.Bar, barSets map[string][]domain.Bar, benchmarkSymbol string) (resolvedSeries, error) {
	haveBars := len(bars) > 0
	haveBarSets := len(barSets) > 0

	if haveBars == haveBarSets {
		if haveBars {
			return resolvedSeries{}, fmt.Errorf("eval: exactly one of Bars / BarSets may be set, got both")
		}
		return resolvedSeries{}, fmt.Errorf("eval: exactly one of Bars / BarSets must be set, got neither")
	}

	if haveBars {
		return resolvedSeries{
			n:      len(bars),
			multi:  false,
			timeAt: func(i int) time.Time { return bars[i].Time },
			windowConfig: func(a, b int) backtest.Config {
				return backtest.Config{Bars: bars[a:b]}
			},
			benchSymbol:        "",
			benchBarsForWindow: func(a, b int) []domain.Bar { return bars[a:b] },
			fullBenchBars:      func() []domain.Bar { return bars },
		}, nil
	}

	// Multi-symbol path: the factory must produce a MultiSymbol strategy, or
	// every fold's train backtest.Run call would fail identically anyway —
	// checking once up front gives a much clearer error than "fold 1: no
	// candidate could be backtested".
	proto := factory()
	ms, ok := proto.(strategy.MultiSymbol)
	if !ok {
		return resolvedSeries{}, fmt.Errorf("eval: strategy %q does not implement MultiSymbol; BarSets requires one", proto.Name())
	}

	// The clock and every fold window are built over EXACTLY the strategy's
	// declared universe, never over all of barSets: the engine clocks its own
	// run over Universe() only, so a barSets key outside the universe whose
	// calendar contributes extra ticks would make eval's tick indices diverge
	// from the engine's — silently misaligning every fold's OOS slicing.
	// Extra barSets keys remain legal (they may serve as the benchmark), they
	// just can never influence the clock or reach the engine.
	universe := ms.Universe()
	clockSets := make(map[string][]domain.Bar, len(universe))
	var missingSyms []string
	for _, sym := range universe {
		series, ok := barSets[sym]
		if !ok || len(series) == 0 {
			missingSyms = append(missingSyms, sym)
			continue
		}
		clockSets[sym] = series
	}
	if len(missingSyms) > 0 {
		return resolvedSeries{}, fmt.Errorf("eval: BarSets is missing universe symbol(s) %v required by strategy %q", missingSyms, proto.Name())
	}

	clock := backtest.MasterClock(clockSets)
	n := len(clock)

	bench := benchmarkSymbol
	if bench == "" {
		if _, ok := barSets["SPY"]; ok {
			bench = "SPY"
		} else {
			for sym := range barSets {
				if bench == "" || sym < bench {
					bench = sym
				}
			}
		}
	}
	benchBars, ok := barSets[bench]
	if !ok {
		return resolvedSeries{}, fmt.Errorf("eval: benchmark symbol %q is not present in BarSets", bench)
	}

	// benchAt indexes the benchmark's own bars by time for O(1) lookup
	// during the OOS-coverage check and window slicing.
	benchAt := make(map[time.Time]int, len(benchBars))
	for i, b := range benchBars {
		benchAt[b.Time] = i
	}

	rs := resolvedSeries{
		n:      n,
		multi:  true,
		timeAt: func(i int) time.Time { return clock[i] },
		windowConfig: func(a, b int) backtest.Config {
			// clockSets, not barSets: the window must contain exactly the
			// symbols whose bars define the clock (see above).
			return backtest.Config{BarSets: backtest.WindowBarSets(clockSets, clock[a], clock[b-1])}
		},
		benchSymbol: bench,
		benchBarsForWindow: func(a, b int) []domain.Bar {
			start, ok := benchAt[clock[a]]
			if !ok {
				return nil
			}
			// clock[b-1] is the window's last tick; find the benchmark bar
			// index at or before it via the same map (benchmark coverage is
			// validated by validateBenchmarkCoverage before this is ever
			// called on the OOS region, so the direct lookup is safe there;
			// callers windowing a train region tolerate a shorter result).
			end, ok := benchAt[clock[b-1]]
			if !ok {
				// Fall back to a scan for the nearest earlier index — only
				// reachable for windows this package doesn't itself request
				// on the OOS side (guarded separately).
				for i := b - 2; i >= a; i-- {
					if idx, ok := benchAt[clock[i]]; ok {
						end = idx
						break
					}
				}
			}
			if end < start {
				return nil
			}
			return benchBars[start : end+1]
		},
		fullBenchBars: func() []domain.Bar { return benchBars },
	}

	return rs, nil
}

// validateBenchmarkCoverage checks that the benchmark symbol has a bar at
// EVERY master-clock tick in [a, b) (the OOS region), returning a
// descriptive error naming the miss count if not. Only meaningful (and only
// called) for a multi-symbol resolvedSeries.
func (rs resolvedSeries) validateBenchmarkCoverage(barSets map[string][]domain.Bar, a, b int) error {
	benchBars := barSets[rs.benchSymbol]
	have := make(map[time.Time]struct{}, len(benchBars))
	for _, bar := range benchBars {
		have[bar.Time] = struct{}{}
	}

	misses := 0
	for i := a; i < b; i++ {
		if _, ok := have[rs.timeAt(i)]; !ok {
			misses++
		}
	}
	if misses > 0 {
		return fmt.Errorf("eval: benchmark symbol %s is missing %d of %d OOS ticks; pass a benchmark that trades every session (e.g. the earliest-listed universe symbol)",
			rs.benchSymbol, misses, b-a)
	}
	return nil
}
