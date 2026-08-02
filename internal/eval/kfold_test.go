package eval

import (
	"math"
	"reflect"
	"testing"

	"tradeforge/internal/strategy"
)

func TestKFoldSpan(t *testing.T) {
	tests := []struct {
		n, k, i      int
		wantS, wantE int
	}{
		// n=10,k=3: [0,3),[3,6),[6,10)
		{10, 3, 0, 0, 3},
		{10, 3, 1, 3, 6},
		{10, 3, 2, 6, 10},
		// n=9,k=3: exact thirds
		{9, 3, 0, 0, 3},
		{9, 3, 1, 3, 6},
		{9, 3, 2, 6, 9},
	}
	for _, tc := range tests {
		s, e := kfoldSpan(tc.n, tc.k, tc.i)
		if s != tc.wantS || e != tc.wantE {
			t.Errorf("kfoldSpan(%d,%d,%d) = (%d,%d), want (%d,%d)", tc.n, tc.k, tc.i, s, e, tc.wantS, tc.wantE)
		}
	}
}

func TestKFoldSpan_CoversWithoutGapsOrOverlap(t *testing.T) {
	cases := []struct{ n, k int }{
		{10, 3}, {9, 3}, {100, 7}, {17, 5}, {2, 2}, {50, 4},
	}
	for _, c := range cases {
		prevEnd := 0
		for i := 0; i < c.k; i++ {
			s, e := kfoldSpan(c.n, c.k, i)
			if s != prevEnd {
				t.Errorf("n=%d k=%d: fold %d start=%d, want %d (gap or overlap)", c.n, c.k, i, s, prevEnd)
			}
			if e < s {
				t.Errorf("n=%d k=%d: fold %d end=%d < start=%d", c.n, c.k, i, e, s)
			}
			prevEnd = e
		}
		if prevEnd != c.n {
			t.Errorf("n=%d k=%d: last fold end=%d, want %d", c.n, c.k, prevEnd, c.n)
		}
	}
}

func TestKFoldTrainSegments(t *testing.T) {
	// n=100, k=5 -> folds: [0,20) [20,40) [40,60) [60,80) [80,100)
	n, k := 100, 5

	// Fold 0: no left segment. testStart=0, testEnd=20. purge=5, embargo=3.
	// right = [min(100, 20+5+3), 100) = [28, 100)
	segs := kfoldTrainSegments(n, k, 0, 5, 3)
	want := [][2]int{{28, 100}}
	if !reflect.DeepEqual(segs, want) {
		t.Errorf("fold 0: segs = %v, want %v", segs, want)
	}

	// Last fold (i=4): testStart=80, testEnd=100. No right segment (embargo
	// pushes past n, but rightStart >= n so it's omitted).
	// left = [0, max(0, 80-5)) = [0, 75)
	segs = kfoldTrainSegments(n, k, 4, 5, 3)
	want = [][2]int{{0, 75}}
	if !reflect.DeepEqual(segs, want) {
		t.Errorf("fold 4 (last): segs = %v, want %v", segs, want)
	}

	// Middle fold (i=2): testStart=40, testEnd=60. purge=5, embargo=3.
	// left  = [0, max(0,40-5)) = [0,35)
	// right = [min(100,60+5+3), 100) = [68,100)
	segs = kfoldTrainSegments(n, k, 2, 5, 3)
	want = [][2]int{{0, 35}, {68, 100}}
	if !reflect.DeepEqual(segs, want) {
		t.Errorf("fold 2 (middle): segs = %v, want %v", segs, want)
	}

	// purge/embargo large enough to make the left segment empty and omitted.
	// Fold 1: testStart=20. purge=25 -> leftEnd = max(0,20-25) = 0 -> omitted.
	segs = kfoldTrainSegments(n, k, 1, 25, 0)
	for _, s := range segs {
		if s[0] == 0 && s[1] == 0 {
			t.Errorf("fold 1 with large purge: empty segment should be omitted, got %v", segs)
		}
	}
	// Confirm no zero-length segment slipped through, and left is indeed gone.
	if len(segs) > 0 && segs[0][0] == 0 {
		// left segment present means leftEnd was > 0; with purge=25 and
		// testStart=20, leftEnd = max(0,-5) = 0, so left must be absent.
		if segs[0][1] == 0 {
			t.Errorf("expected left segment to be omitted, got %v", segs)
		}
	}
}

func TestKFoldTrainSegments_PurgeExcludesAdjacentBars(t *testing.T) {
	// Purge > 0 must exclude bars immediately adjacent to the test fold from
	// every train range.
	n, k, i, purge, embargo := 100, 5, 2, 10, 0
	testStart, testEnd := kfoldSpan(n, k, i)

	segs := kfoldTrainSegments(n, k, i, purge, embargo)
	for _, seg := range segs {
		for bar := testStart - purge; bar < testStart; bar++ {
			if bar >= seg[0] && bar < seg[1] {
				t.Errorf("bar %d (within purge before test fold) unexpectedly included in train segment %v", bar, seg)
			}
		}
		for bar := testEnd; bar < testEnd+purge; bar++ {
			if bar >= seg[0] && bar < seg[1] {
				t.Errorf("bar %d (within purge after test fold) unexpectedly included in train segment %v", bar, seg)
			}
		}
	}

	// With purge=0, adjacent bars ARE allowed back into train (sanity check
	// that the purge mechanism, not some other exclusion, is doing the work).
	segsNoPurge := kfoldTrainSegments(n, k, i, 0, 0)
	found := false
	for _, seg := range segsNoPurge {
		if testStart-1 >= seg[0] && testStart-1 < seg[1] {
			found = true
		}
	}
	if !found {
		t.Error("with purge=0, the bar immediately before the test fold should be in a train segment")
	}
}

func TestKFold_EndToEnd(t *testing.T) {
	bars := risingBars(600, 100)

	res, err := KFold(KFoldConfig{
		Factory:     func() strategy.Strategy { return newSwitchStrategy() },
		Bars:        bars,
		Symbol:      "TEST",
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
		if fold.Trials != 2 {
			t.Errorf("fold %d: Trials = %d, want 2 (bad=1 combos excluded)", fold.Fold, fold.Trials)
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
		t.Fatalf("len(OOSEquity) = %d, want 600 (all folds' full spans)", len(res.OOSEquity))
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

func TestKFold_Determinism(t *testing.T) {
	bars := risingBars(600, 100)

	cfg := KFoldConfig{
		Factory:     func() strategy.Strategy { return newSwitchStrategy() },
		Bars:        bars,
		Symbol:      "TEST",
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
		t.Fatalf("two KFold() runs were not deeply equal:\na=%+v\nb=%+v", a, b)
	}
}

func TestKFold_NonTunableStrategy(t *testing.T) {
	bars := risingBars(600, 100)

	res, err := KFold(KFoldConfig{
		Factory:     func() strategy.Strategy { return newPlainSwitchStrategy() },
		Bars:        bars,
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
		if fold.Trials != 1 {
			t.Errorf("fold %d: Trials = %d, want 1", fold.Fold, fold.Trials)
		}
	}
}

func TestKFold_RequiresFactory(t *testing.T) {
	_, err := KFold(KFoldConfig{Bars: risingBars(600, 100)})
	if err == nil {
		t.Fatal("KFold() error = nil, want an error for nil Factory")
	}
}

func TestKFold_FoldsBelowMinimum(t *testing.T) {
	_, err := KFold(KFoldConfig{
		Factory: func() strategy.Strategy { return newSwitchStrategy() },
		Bars:    risingBars(600, 100),
		Folds:   1,
	})
	if err == nil {
		t.Fatal("KFold() error = nil, want an error for Folds < 2")
	}
}

func TestKFold_TooFewBars(t *testing.T) {
	_, err := KFold(KFoldConfig{
		Factory: func() strategy.Strategy { return newSwitchStrategy() },
		Bars:    risingBars(5, 100),
		Folds:   5, // n/k = 1 < 2
	})
	if err == nil {
		t.Fatal("KFold() error = nil, want an error when n/k < 2")
	}
}

func TestKFold_WarmupGuardTriggers(t *testing.T) {
	// switchStrategy's warmup is 5. With very small data and a large fold
	// count, every train segment will be too short to exceed warmup.
	_, err := KFold(KFoldConfig{
		Factory:     func() strategy.Strategy { return newSwitchStrategy() },
		Bars:        risingBars(20, 100),
		Folds:       10, // n/k = 2 bars/fold; train segments tiny
		PurgeBars:   0,
		EmbargoBars: 0,
		Objective:   totalReturnObjective,
	})
	if err == nil {
		t.Fatal("KFold() error = nil, want an error when every train segment <= warmup")
	}
}

func TestKFold_NegativePurgeRejected(t *testing.T) {
	if _, err := KFold(KFoldConfig{
		Factory:   func() strategy.Strategy { return newSwitchStrategy() },
		Bars:      risingBars(600, 100),
		PurgeBars: -1,
	}); err == nil {
		t.Error("KFold() error = nil, want error for negative PurgeBars")
	}
}

// TestKFold_NegativeEmbargoMeansAuto documents the deliberate asymmetry
// with PurgeBars: EmbargoBars uses a NEGATIVE sentinel (not 0) to request
// the auto default, because 0 is itself a legitimate, literal "no embargo"
// value that the CLI's --embargo 0 must be able to express (see
// KFoldConfig.EmbargoBars's doc). A negative value must therefore succeed,
// not error, and behave the same as EmbargoBars left unset (0 given
// explicitly should differ: 0 requests a literal zero embargo).
func TestKFold_NegativeEmbargoMeansAuto(t *testing.T) {
	bars := risingBars(600, 100)
	cfgAuto := KFoldConfig{
		Factory:     func() strategy.Strategy { return newSwitchStrategy() },
		Bars:        bars,
		Folds:       4,
		PurgeBars:   5,
		EmbargoBars: -1,
		Objective:   totalReturnObjective,
	}
	resAuto, err := KFold(cfgAuto)
	if err != nil {
		t.Fatalf("KFold() with EmbargoBars=-1 error = %v, want success (auto default)", err)
	}
	if resAuto.EmbargoBars != len(bars)/embargoDivisor {
		t.Errorf("EmbargoBars = %d, want auto default %d", resAuto.EmbargoBars, len(bars)/embargoDivisor)
	}

	cfgZero := cfgAuto
	cfgZero.EmbargoBars = 0
	resZero, err := KFold(cfgZero)
	if err != nil {
		t.Fatalf("KFold() with EmbargoBars=0 error = %v, want success (literal zero)", err)
	}
	if resZero.EmbargoBars != 0 {
		t.Errorf("EmbargoBars = %d, want literal 0", resZero.EmbargoBars)
	}
}

func TestKFold_TrialSRVarAndDSRFinite(t *testing.T) {
	bars := risingBars(600, 100)

	res, err := KFold(KFoldConfig{
		Factory:     func() strategy.Strategy { return newSwitchStrategy() },
		Bars:        bars,
		Symbol:      "TEST",
		Folds:       4,
		PurgeBars:   5,
		EmbargoBars: 2,
		Objective:   totalReturnObjective,
	})
	if err != nil {
		t.Fatalf("KFold() error = %v", err)
	}

	if math.IsNaN(res.TrialSRVar) || math.IsInf(res.TrialSRVar, 0) {
		t.Errorf("TrialSRVar = %v, want finite", res.TrialSRVar)
	}
	if math.IsNaN(res.DSR) || math.IsInf(res.DSR, 0) {
		t.Errorf("DSR = %v, want finite", res.DSR)
	}
	if res.TotalTrials < 2 {
		t.Fatalf("TotalTrials = %d, want >= 2", res.TotalTrials)
	}
}
