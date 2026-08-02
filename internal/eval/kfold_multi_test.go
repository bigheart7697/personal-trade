package eval

import (
	"reflect"
	"testing"

	"tradeforge/internal/strategy"
)

// TestKFoldMulti_EndToEnd runs purged & embargoed K-fold on a two-symbol
// universe end to end: fold count, OOS length == n (master-clock ticks),
// ParamWins populated, and sane DSR/benchmark output.
func TestKFoldMulti_EndToEnd(t *testing.T) {
	barSets := twoSymbolBarSets(600, 0)

	res, err := KFold(KFoldConfig{
		Factory:     func() strategy.Strategy { return newSwitchMultiStrategy([]string{"A", "B"}) },
		BarSets:     barSets,
		Symbol:      "A+B",
		Folds:       4,
		PurgeBars:   5,
		EmbargoBars: 2,
		Objective:   totalReturnObjective,
	})
	if err != nil {
		t.Fatalf("KFold() error = %v", err)
	}

	if len(res.FoldResults) != 4 {
		t.Fatalf("len(FoldResults) = %d, want 4", len(res.FoldResults))
	}

	for _, fold := range res.FoldResults {
		if fold.BestParams == nil {
			t.Fatalf("fold %d: BestParams = nil, want non-nil", fold.Fold)
		}
		if fold.BestParams["on"] != 1 {
			t.Errorf("fold %d: BestParams[on] = %v, want 1 (on beats off in a rising market)", fold.Fold, fold.BestParams["on"])
		}
	}

	if len(res.ParamWins) != 1 {
		t.Fatalf("len(ParamWins) = %d, want 1 distinct winning combo", len(res.ParamWins))
	}
	for combo, wins := range res.ParamWins {
		if wins != 4 {
			t.Errorf("ParamWins[%q] = %d, want 4", combo, wins)
		}
	}

	if len(res.OOSEquity) != 600 {
		t.Fatalf("len(OOSEquity) = %d, want 600 (all folds' full tick spans)", len(res.OOSEquity))
	}
	if res.OOSEquity[0].Equity != DefaultInitialCash {
		t.Errorf("OOSEquity[0].Equity = %v, want %v", res.OOSEquity[0].Equity, DefaultInitialCash)
	}
	for i, pt := range res.OOSEquity {
		if pt.Equity <= 0 {
			t.Errorf("OOSEquity[%d].Equity = %v, want > 0", i, pt.Equity)
		}
	}

	if res.DSROk {
		if res.DSR < 0 || res.DSR > 1 {
			t.Errorf("DSR = %v, want in [0,1]", res.DSR)
		}
	} else if res.DSR != 0 {
		t.Errorf("DSR = %v, want 0 when !DSROk", res.DSR)
	}

	if res.BenchmarkMetrics.TotalReturn <= 0 {
		t.Errorf("BenchmarkMetrics.TotalReturn = %v, want > 0 on rising data", res.BenchmarkMetrics.TotalReturn)
	}
}

// TestKFoldMulti_Determinism verifies two identical multi-symbol KFold runs
// produce deeply equal results.
func TestKFoldMulti_Determinism(t *testing.T) {
	cfg := KFoldConfig{
		Factory:     func() strategy.Strategy { return newSwitchMultiStrategy([]string{"A", "B"}) },
		BarSets:     twoSymbolBarSets(600, 0),
		Symbol:      "A+B",
		Folds:       4,
		PurgeBars:   5,
		EmbargoBars: 2,
		Objective:   totalReturnObjective,
	}

	a, err := KFold(cfg)
	if err != nil {
		t.Fatalf("KFold() [a] error = %v", err)
	}
	b, err := KFold(cfg)
	if err != nil {
		t.Fatalf("KFold() [b] error = %v", err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("two multi-symbol KFold() runs were not deeply equal:\na=%+v\nb=%+v", a, b)
	}
}

// TestKFoldMulti_EquityCurveLengthInvariant verifies the stitched OOS curve
// covers the full master clock exactly once (the same invariant walk-forward
// checks): K-fold's spans partition [0,n) completely, so windowed segments
// must sum to exactly n ticks.
func TestKFoldMulti_EquityCurveLengthInvariant(t *testing.T) {
	barSets := twoSymbolBarSets(600, 0)

	res, err := KFold(KFoldConfig{
		Factory:     func() strategy.Strategy { return newSwitchMultiStrategy([]string{"A", "B"}) },
		BarSets:     barSets,
		Symbol:      "A+B",
		Folds:       5,
		PurgeBars:   5,
		EmbargoBars: 2,
		Objective:   totalReturnObjective,
	})
	if err != nil {
		t.Fatalf("KFold() error = %v", err)
	}
	if len(res.OOSEquity) != 600 {
		t.Fatalf("len(OOSEquity) = %d, want 600 (n ticks, partitioned without gaps)", len(res.OOSEquity))
	}
}

// TestKFoldMulti_NonTunableStrategy exercises the non-Tunable multi-symbol
// path.
func TestKFoldMulti_NonTunableStrategy(t *testing.T) {
	res, err := KFold(KFoldConfig{
		Factory:     func() strategy.Strategy { return newPlainSwitchMultiStrategy([]string{"A", "B"}) },
		BarSets:     twoSymbolBarSets(600, 0),
		Folds:       4,
		PurgeBars:   5,
		EmbargoBars: 2,
		Objective:   totalReturnObjective,
	})
	if err != nil {
		t.Fatalf("KFold() error = %v", err)
	}
	if len(res.ParamWins) != 0 {
		t.Errorf("ParamWins = %+v, want empty for non-tunable strategy", res.ParamWins)
	}
	for _, fold := range res.FoldResults {
		if fold.BestParams != nil {
			t.Errorf("fold %d: BestParams = %+v, want nil (non-Tunable strategy)", fold.Fold, fold.BestParams)
		}
	}
}

// TestKFoldMulti_BenchmarkMissingTicksErrors verifies the documented
// benchmark-coverage error fires for K-fold too, over the full [0,n) OOS
// region (K-fold's folds partition the entire clock).
func TestKFoldMulti_BenchmarkMissingTicksErrors(t *testing.T) {
	barSets := twoSymbolBarSets(600, 150) // B lists 150 ticks after A

	_, err := KFold(KFoldConfig{
		Factory:         func() strategy.Strategy { return newSwitchMultiStrategy([]string{"A", "B"}) },
		BarSets:         barSets,
		Symbol:          "A+B",
		Folds:           4,
		PurgeBars:       5,
		EmbargoBars:     2,
		BenchmarkSymbol: "B",
		Objective:       totalReturnObjective,
	})
	if err == nil {
		t.Fatal("KFold() error = nil, want the benchmark-missing-ticks error")
	}
	if got := err.Error(); !contains(got, "benchmark symbol B is missing") {
		t.Errorf("error = %q, want it to mention \"benchmark symbol B is missing\"", got)
	}
}

// TestKFoldMulti_BarsAndBarSetsBothSet verifies KFoldConfig validation
// rejects a run with both Bars and BarSets set.
func TestKFoldMulti_BarsAndBarSetsBothSet(t *testing.T) {
	_, err := KFold(KFoldConfig{
		Factory: func() strategy.Strategy { return newSwitchMultiStrategy([]string{"A", "B"}) },
		Bars:    risingBars(600, 100),
		BarSets: twoSymbolBarSets(600, 0),
	})
	if err == nil {
		t.Fatal("KFold() error = nil, want an error when both Bars and BarSets are set")
	}
}

// TestKFoldMulti_NonMultiSymbolStrategyWithBarSets verifies the resolution
// helper pre-validates BarSets requires a MultiSymbol strategy.
func TestKFoldMulti_NonMultiSymbolStrategyWithBarSets(t *testing.T) {
	_, err := KFold(KFoldConfig{
		Factory: func() strategy.Strategy { return newSwitchStrategy() }, // not MultiSymbol
		BarSets: twoSymbolBarSets(600, 0),
	})
	if err == nil {
		t.Fatal("KFold() error = nil, want an error when a non-MultiSymbol strategy is given BarSets")
	}
	if got := err.Error(); !contains(got, "does not implement MultiSymbol") {
		t.Errorf("error = %q, want it to mention \"does not implement MultiSymbol\"", got)
	}
}

// TestKFoldMulti_EmbargoAutoDefaultInTicks verifies EmbargoBars' auto
// default (-1) divides the master-clock tick count, not a bar count from
// some other series.
func TestKFoldMulti_EmbargoAutoDefaultInTicks(t *testing.T) {
	barSets := twoSymbolBarSets(600, 0)

	res, err := KFold(KFoldConfig{
		Factory:     func() strategy.Strategy { return newSwitchMultiStrategy([]string{"A", "B"}) },
		BarSets:     barSets,
		Folds:       4,
		PurgeBars:   5,
		EmbargoBars: -1,
		Objective:   totalReturnObjective,
	})
	if err != nil {
		t.Fatalf("KFold() error = %v", err)
	}
	wantEmbargo := 600 / embargoDivisor
	if res.EmbargoBars != wantEmbargo {
		t.Errorf("EmbargoBars = %d, want %d (600 master-clock ticks / %d)", res.EmbargoBars, wantEmbargo, embargoDivisor)
	}
}
