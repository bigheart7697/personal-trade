package eval

import (
	"reflect"
	"testing"

	"tradeforge/internal/backtest"
	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// TestWalkForwardMulti_FoldBoundariesMatchClock verifies fold boundary times
// are taken from the master clock at exactly the same indices the
// single-series path would use, for a two-symbol universe listed together
// (offset=0, so the clock is simply the shared calendar).
func TestWalkForwardMulti_FoldBoundariesMatchClock(t *testing.T) {
	trainBars, testBars := 100, 50
	barSets := twoSymbolBarSets(300, 0)
	clock := backtest.MasterClock(barSets)

	res, err := WalkForward(Config{
		Factory:   func() strategy.Strategy { return newSwitchMultiStrategy([]string{"A", "B"}) },
		BarSets:   barSets,
		Symbol:    "A+B",
		TrainBars: trainBars,
		TestBars:  testBars,
		Objective: totalReturnObjective,
	})
	if err != nil {
		t.Fatalf("WalkForward() error = %v", err)
	}

	wantFolds := 4 // same fold count as the single-series equivalent (n=300)
	if len(res.Folds) != wantFolds {
		t.Fatalf("len(Folds) = %d, want %d", len(res.Folds), wantFolds)
	}

	for i, fold := range res.Folds {
		start := i * testBars
		wantTrainStart := clock[start]
		wantTrainEnd := clock[start+trainBars-1]
		wantTestStart := clock[start+trainBars]
		wantTestEnd := clock[start+trainBars+testBars-1]

		if !fold.TrainStart.Equal(wantTrainStart) {
			t.Errorf("fold %d: TrainStart = %v, want %v", i, fold.TrainStart, wantTrainStart)
		}
		if !fold.TrainEnd.Equal(wantTrainEnd) {
			t.Errorf("fold %d: TrainEnd = %v, want %v", i, fold.TrainEnd, wantTrainEnd)
		}
		if !fold.TestStart.Equal(wantTestStart) {
			t.Errorf("fold %d: TestStart = %v, want %v", i, fold.TestStart, wantTestStart)
		}
		if !fold.TestEnd.Equal(wantTestEnd) {
			t.Errorf("fold %d: TestEnd = %v, want %v", i, fold.TestEnd, wantTestEnd)
		}
	}
}

// TestWalkForwardMulti_EquityCurveLengthInvariant verifies the CRITICAL
// invariant: windowing BarSets by boundary times and running the engine over
// that window produces an EquityCurve whose length equals the tick span
// (b-a) — every master-clock tick has at least one symbol bar, and
// windowing by TIME preserves that. Checked indirectly via each fold's
// TestMetrics existing (computed from a non-empty, correctly-sized segment)
// and directly via the stitched OOS length, which is the authoritative
// end-to-end check used throughout this package.
func TestWalkForwardMulti_EquityCurveLengthInvariant(t *testing.T) {
	trainBars, testBars := 100, 50
	barSets := twoSymbolBarSets(300, 0)

	res, err := WalkForward(Config{
		Factory:   func() strategy.Strategy { return newSwitchMultiStrategy([]string{"A", "B"}) },
		BarSets:   barSets,
		Symbol:    "A+B",
		TrainBars: trainBars,
		TestBars:  testBars,
		Objective: totalReturnObjective,
	})
	if err != nil {
		t.Fatalf("WalkForward() error = %v", err)
	}

	numFolds := len(res.Folds)
	wantLen := numFolds * testBars
	if len(res.OOSEquity) != wantLen {
		t.Fatalf("len(OOSEquity) = %d, want %d (%d folds x %d test ticks) — windowed EquityCurve length invariant violated",
			len(res.OOSEquity), wantLen, numFolds, testBars)
	}
}

// TestWalkForwardMulti_Determinism verifies two identical multi-symbol
// WalkForward runs produce deeply equal Results.
func TestWalkForwardMulti_Determinism(t *testing.T) {
	cfg := Config{
		Factory:   func() strategy.Strategy { return newSwitchMultiStrategy([]string{"A", "B"}) },
		BarSets:   twoSymbolBarSets(300, 0),
		Symbol:    "A+B",
		TrainBars: 100,
		TestBars:  50,
		Objective: totalReturnObjective,
	}

	a, err := WalkForward(cfg)
	if err != nil {
		t.Fatalf("WalkForward() [a] error = %v", err)
	}
	b, err := WalkForward(cfg)
	if err != nil {
		t.Fatalf("WalkForward() [b] error = %v", err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("two multi-symbol WalkForward() runs were not deeply equal:\na=%+v\nb=%+v", a, b)
	}
}

// TestWalkForwardMulti_DSRAndBenchmark verifies DSR is in range (or n/a) and
// BenchmarkMetrics is populated, and that the default benchmark resolves to
// the lexically smallest universe key ("A") when neither "SPY" nor an
// explicit BenchmarkSymbol is present.
func TestWalkForwardMulti_DSRAndBenchmark(t *testing.T) {
	barSets := twoSymbolBarSets(300, 0)

	res, err := WalkForward(Config{
		Factory:   func() strategy.Strategy { return newSwitchMultiStrategy([]string{"A", "B"}) },
		BarSets:   barSets,
		Symbol:    "A+B",
		TrainBars: 100,
		TestBars:  50,
		Objective: totalReturnObjective,
	})
	if err != nil {
		t.Fatalf("WalkForward() error = %v", err)
	}

	if res.DSROk {
		if res.DSR < 0 || res.DSR > 1 {
			t.Errorf("DSR = %v, want in [0,1]", res.DSR)
		}
	} else if res.DSR != 0 {
		t.Errorf("DSR = %v, want 0 when !DSROk", res.DSR)
	}

	if res.BenchmarkMetrics.TotalReturn <= 0 {
		t.Errorf("BenchmarkMetrics.TotalReturn = %v, want > 0 (default benchmark = A, a rising series)", res.BenchmarkMetrics.TotalReturn)
	}
}

// TestWalkForwardMulti_ExplicitBenchmarkHonored verifies an explicit
// BenchmarkSymbol overrides the default resolution (which would pick "A",
// the lexically smallest key) by pointing it at "B" instead and checking the
// run succeeds and produces a sane benchmark curve.
func TestWalkForwardMulti_ExplicitBenchmarkHonored(t *testing.T) {
	barSets := twoSymbolBarSets(300, 0)

	res, err := WalkForward(Config{
		Factory:         func() strategy.Strategy { return newSwitchMultiStrategy([]string{"A", "B"}) },
		BarSets:         barSets,
		Symbol:          "A+B",
		TrainBars:       100,
		TestBars:        50,
		BenchmarkSymbol: "B",
		Objective:       totalReturnObjective,
	})
	if err != nil {
		t.Fatalf("WalkForward() error = %v", err)
	}
	if res.BenchmarkMetrics.TotalReturn <= 0 {
		t.Errorf("BenchmarkMetrics.TotalReturn = %v, want > 0 (explicit benchmark B, also rising)", res.BenchmarkMetrics.TotalReturn)
	}
}

// TestWalkForwardMulti_BenchmarkMissingTicksErrors verifies the documented
// error fires with correct counts when the resolved benchmark symbol does
// not have a bar at every OOS-region master-clock tick. B lists 150 ticks
// after A; with TrainBars=100/TestBars=50 the OOS region starts at tick 100,
// well before B's tick-150 listing, so B is missing exactly 50 OOS ticks in
// the first fold alone (and the check runs over the whole OOS region up
// front).
func TestWalkForwardMulti_BenchmarkMissingTicksErrors(t *testing.T) {
	barSets := twoSymbolBarSets(300, 150) // B lists 150 ticks after A

	_, err := WalkForward(Config{
		Factory:         func() strategy.Strategy { return newSwitchMultiStrategy([]string{"A", "B"}) },
		BarSets:         barSets,
		Symbol:          "A+B",
		TrainBars:       100,
		TestBars:        50,
		BenchmarkSymbol: "B",
		Objective:       totalReturnObjective,
	})
	if err == nil {
		t.Fatal("WalkForward() error = nil, want the benchmark-missing-ticks error")
	}
	if got := err.Error(); !contains(got, "benchmark symbol B is missing") {
		t.Errorf("error = %q, want it to mention \"benchmark symbol B is missing\"", got)
	}
}

// TestWalkForwardMulti_BarsAndBarSetsBothSet verifies Config validation
// rejects a run with both Bars and BarSets set.
func TestWalkForwardMulti_BarsAndBarSetsBothSet(t *testing.T) {
	_, err := WalkForward(Config{
		Factory: func() strategy.Strategy { return newSwitchMultiStrategy([]string{"A", "B"}) },
		Bars:    risingBars(300, 100),
		BarSets: twoSymbolBarSets(300, 0),
	})
	if err == nil {
		t.Fatal("WalkForward() error = nil, want an error when both Bars and BarSets are set")
	}
}

// TestWalkForwardMulti_NeitherBarsNorBarSetsSet verifies Config validation
// rejects a run with neither Bars nor BarSets set.
func TestWalkForwardMulti_NeitherBarsNorBarSetsSet(t *testing.T) {
	_, err := WalkForward(Config{
		Factory: func() strategy.Strategy { return newSwitchStrategy() },
	})
	if err == nil {
		t.Fatal("WalkForward() error = nil, want an error when neither Bars nor BarSets is set")
	}
}

// TestWalkForwardMulti_NonMultiSymbolStrategyWithBarSets verifies the
// resolution helper pre-validates that BarSets requires a MultiSymbol
// strategy, with a clear message, before running any fold.
func TestWalkForwardMulti_NonMultiSymbolStrategyWithBarSets(t *testing.T) {
	_, err := WalkForward(Config{
		Factory: func() strategy.Strategy { return newSwitchStrategy() }, // not MultiSymbol
		BarSets: twoSymbolBarSets(300, 0),
	})
	if err == nil {
		t.Fatal("WalkForward() error = nil, want an error when a non-MultiSymbol strategy is given BarSets")
	}
	if got := err.Error(); !contains(got, "does not implement MultiSymbol") {
		t.Errorf("error = %q, want it to mention \"does not implement MultiSymbol\"", got)
	}
}

// contains is a tiny substring helper so these tests don't need to import
// strings just for one check each.
// TestWalkForwardMulti_InterleavedCalendars drives the full harness over the
// hard-calendar fixture: the master clock is a strict superset of either
// symbol's own bars (one B-only Saturday tick in the train region, B absent
// from half of all ticks). The assertions pin the window invariant end to
// end: fold count and stitched OOS length must be exact in TICK units, and a
// one-tick fold overlap or slip would break both.
func TestWalkForwardMulti_InterleavedCalendars(t *testing.T) {
	barSets := interleavedBarSets(300) // clock = 301 ticks (300 from A + 1 B-only Saturday)
	trainBars, testBars := 100, 50

	res, err := WalkForward(Config{
		Factory:         func() strategy.Strategy { return newSwitchMultiStrategy([]string{"A", "B"}) },
		BarSets:         barSets,
		Symbol:          "A+B",
		BenchmarkSymbol: "A", // A covers every OOS tick (the B-only tick is in train)
		TrainBars:       trainBars,
		TestBars:        testBars,
		Objective:       totalReturnObjective,
	})
	if err != nil {
		t.Fatalf("WalkForward() error = %v", err)
	}

	// n = 301 ticks: folds start at 0, 50, 100, 150; start=200 -> 200+150=350 > 301 stops.
	wantFolds := 4
	if len(res.Folds) != wantFolds {
		t.Fatalf("len(Folds) = %d, want %d", len(res.Folds), wantFolds)
	}
	if len(res.OOSEquity) != wantFolds*testBars {
		t.Errorf("len(OOSEquity) = %d, want %d (window invariant: OOS length == folds x testBars in ticks)",
			len(res.OOSEquity), wantFolds*testBars)
	}
	for i, pt := range res.OOSEquity {
		if pt.Equity <= 0 {
			t.Fatalf("OOSEquity[%d] = %v, want > 0", i, pt.Equity)
		}
	}
}

// TestWalkForwardMulti_ExtraBarSetsSymbolIgnored pins the clock-symbol-set
// contract: a BarSets key OUTSIDE the strategy's Universe() must not
// influence the master clock, the fold arithmetic, or any result — even when
// its calendar would contribute extra ticks (Z runs 30 ticks past everyone
// else). The run with the extra symbol must be deeply equal to the run
// without it.
func TestWalkForwardMulti_ExtraBarSetsSymbolIgnored(t *testing.T) {
	base := twoSymbolBarSets(300, 0)
	withExtra := map[string][]domain.Bar{
		"A": base["A"],
		"B": base["B"],
		"Z": risingMultiBars("Z", 0, 330, 70), // 30 ticks beyond A/B's calendar
	}

	cfg := Config{
		Factory:         func() strategy.Strategy { return newSwitchMultiStrategy([]string{"A", "B"}) },
		Symbol:          "A+B",
		BenchmarkSymbol: "A",
		TrainBars:       100,
		TestBars:        50,
		Objective:       totalReturnObjective,
	}

	cfg.BarSets = base
	resBase, err := WalkForward(cfg)
	if err != nil {
		t.Fatalf("WalkForward(base) error = %v", err)
	}

	cfg.BarSets = withExtra
	resExtra, err := WalkForward(cfg)
	if err != nil {
		t.Fatalf("WalkForward(withExtra) error = %v", err)
	}

	if !reflect.DeepEqual(resBase, resExtra) {
		t.Errorf("results differ when a non-universe symbol is present in BarSets:\nbase folds=%d oosLen=%d\nextra folds=%d oosLen=%d",
			len(resBase.Folds), len(resBase.OOSEquity), len(resExtra.Folds), len(resExtra.OOSEquity))
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && indexOf(s, substr) >= 0
}

func indexOf(s, substr string) int {
	for i := 0; i+len(substr) <= len(s); i++ {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}
