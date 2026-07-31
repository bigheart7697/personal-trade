package strategies

import (
	"testing"

	"tradeforge/internal/strategy"
)

func TestMLBoost_Determinism(t *testing.T) {
	bars := scBarsFromCloses(mlRegimeCloses(8, 40))

	s1 := &mlBoost{trainWindow: 100, deadband: 0.0, refitEvery: 63}
	s2 := &mlBoost{trainWindow: 100, deadband: 0.0, refitEvery: 63}

	sigs1 := runMLStrategy(s1, bars[:260])
	sigs2 := runMLStrategy(s2, bars[:260])

	if len(sigs1) != len(sigs2) {
		t.Fatalf("signal sequence lengths differ: %d vs %d", len(sigs1), len(sigs2))
	}
	for i := range sigs1 {
		if !signalsEqual(sigs1[i], sigs2[i]) {
			t.Fatalf("signal at index %d differs between two fresh instances fed identical bars: %+v vs %+v", i, sigs1[i], sigs2[i])
		}
	}
}

// TestMLBoost_FreshInstanceMatchesLongLived: same backtest/paper
// equivalence property as TestMLLogit_FreshInstanceMatchesLongLived (see
// its doc comment) — the governing model must be a pure function of
// history, not process lifetime.
func TestMLBoost_FreshInstanceMatchesLongLived(t *testing.T) {
	bars := scBarsFromCloses(mlRegimeCloses(10, 40))
	port := flatPortfolio()

	long := &mlBoost{trainWindow: 100, deadband: 0.0, refitEvery: 63}
	sawSignal := false
	for i := long.WarmupBars(); i < len(bars); i++ {
		ctx := strategy.NewContext(bars[:i+1], port)
		longSigs := long.OnBar(ctx, bars[i])

		fresh := &mlBoost{trainWindow: 100, deadband: 0.0, refitEvery: 63}
		freshSigs := fresh.OnBar(strategy.NewContext(bars[:i+1], port), bars[i])

		if !signalsEqual(longSigs, freshSigs) {
			t.Fatalf("bar %d: long-lived instance emitted %+v but a fresh one-shot instance emitted %+v — refit timing depends on process lifetime", i, longSigs, freshSigs)
		}
		if len(longSigs) > 0 {
			sawSignal = true
		}
	}
	if !sawSignal {
		t.Fatal("no signal emitted across the whole run; the equivalence comparison never exercised a fitted model (vacuous test)")
	}
}

// TestMLBoostFeatures_ZeroCloseGuard mirrors ml-logit's guard test: a 0.0
// close inside a momentum window invalidates the feature vector.
func TestMLBoostFeatures_ZeroCloseGuard(t *testing.T) {
	closes := make([]float64, 120)
	for i := range closes {
		closes[i] = 100
	}
	closes[99] = 0 // i-1 for i=100
	if _, ok := mlBoostFeatures(closes, 100); ok {
		t.Fatal("mlBoostFeatures ok = true with a zero close in the momentum window, want false")
	}
}

func TestMLBoost_NoLookahead(t *testing.T) {
	bars := scBarsFromCloses(mlRegimeCloses(8, 40))

	n := 200
	sShort := &mlBoost{trainWindow: 100, deadband: 0.0, refitEvery: 63}
	sLong := &mlBoost{trainWindow: 100, deadband: 0.0, refitEvery: 63}

	sigsShort := runMLStrategy(sShort, bars[:n])
	sigsLong := runMLStrategy(sLong, bars[:n+50])

	if len(sigsLong) < len(sigsShort) {
		t.Fatalf("longer run produced fewer signals (%d) than the shorter run (%d)", len(sigsLong), len(sigsShort))
	}
	for i := range sigsShort {
		if !signalsEqual(sigsShort[i], sigsLong[i]) {
			t.Fatalf("signal at index %d changed when 50 future bars were appended: %+v (short) vs %+v (long) — lookahead leak", i, sigsShort[i], sigsLong[i])
		}
	}
}

// TestMLBoost_Learnability reuses the alternating up/down regime series
// from ml_logit_test.go. The boosted stumps are fit on next-bar RETURN
// (not sign), so the same up/down blocks give a clean, learnable
// regression target: once past warmup, the boosted score should be
// positive on a bar deep inside an up block and non-positive on a bar deep
// inside a down block.
func TestMLBoost_Learnability(t *testing.T) {
	closes := mlRegimeCloses(8, 40)
	bars := scBarsFromCloses(closes)

	// Both cases sit exactly ON a refit boundary (length % refitEvery == 0)
	// so the governing model is freshly fit — the learnability property
	// belongs to the fitted model, not to the between-boundary hold, which
	// TestMLBoost_FreshInstanceMatchesLongLived covers separately.

	t.Run("long after an up day", func(t *testing.T) {
		// Length 189 = 3*63: bar 188 sits inside indices 161..200, an up
		// block.
		s := &mlBoost{trainWindow: 100, deadband: 0.0, refitEvery: 63}
		history := bars[:189]
		ctx := strategy.NewContext(history, flatPortfolio())

		r1 := closes[188]/closes[187] - 1
		if r1 <= 0 {
			t.Fatalf("test setup invalid: bar 188's own return = %v, want > 0", r1)
		}

		weights := s.TargetWeights(ctx)
		if got, want := weights["SPY"], 1.0; got != want {
			t.Errorf("TargetWeights()[SPY] = %v, want %v (up-day feature should predict a positive boosted score)", got, want)
		}
	})

	t.Run("flat after a down day", func(t *testing.T) {
		// Length 315 = 5*63: bar 314 sits inside indices 281..320, a down
		// block.
		s := &mlBoost{trainWindow: 100, deadband: 0.0, refitEvery: 63}
		history := bars[:315]
		ctx := strategy.NewContext(history, flatPortfolio())

		r1 := closes[314]/closes[313] - 1
		if r1 >= 0 {
			t.Fatalf("test setup invalid: bar 314's own return = %v, want < 0", r1)
		}

		weights := s.TargetWeights(ctx)
		if len(weights) != 0 {
			t.Errorf("TargetWeights() = %+v, want empty map (down-day feature should not predict a positive boosted score)", weights)
		}
	})
}

func TestMLBoost_TargetWeights_PurityFromPortfolio(t *testing.T) {
	bars := scBarsFromCloses(mlRegimeCloses(8, 40))[:200]

	s1 := &mlBoost{trainWindow: 100, deadband: 0.0, refitEvery: 63}
	s2 := &mlBoost{trainWindow: 100, deadband: 0.0, refitEvery: 63}

	ctxFlat := strategy.NewContext(bars, flatPortfolio())
	ctxLong := strategy.NewContext(bars, longPortfolio(50, 100))

	got1 := s1.TargetWeights(ctxFlat)
	got2 := s2.TargetWeights(ctxLong)

	if len(got1) != len(got2) {
		t.Fatalf("TargetWeights differ by portfolio state: flat=%+v long=%+v", got1, got2)
	}
	for sym, w := range got1 {
		if got2[sym] != w {
			t.Errorf("TargetWeights()[%s] = %v with flat portfolio, %v with a long portfolio; must be identical", sym, w, got2[sym])
		}
	}
}

func TestMLBoost_WithParams(t *testing.T) {
	t.Run("valid full override applied", func(t *testing.T) {
		base := newMLBoost()
		got, err := base.WithParams(map[string]float64{"trainWindow": 756, "deadband": 0.0005})
		if err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		mb, ok := got.(*mlBoost)
		if !ok {
			t.Fatalf("WithParams() returned %T, want *mlBoost", got)
		}
		if mb.trainWindow != 756 || mb.deadband != 0.0005 {
			t.Errorf("trainWindow/deadband = %d/%v, want 756/0.0005", mb.trainWindow, mb.deadband)
		}
	})

	t.Run("partial map keeps defaults", func(t *testing.T) {
		base := newMLBoost()
		got, err := base.WithParams(map[string]float64{"deadband": 0.0005})
		if err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		mb := got.(*mlBoost)
		if mb.deadband != 0.0005 {
			t.Errorf("deadband = %v, want 0.0005", mb.deadband)
		}
		if mb.trainWindow != 504 {
			t.Errorf("trainWindow = %d, want default 504", mb.trainWindow)
		}
	})

	t.Run("unknown key rejected", func(t *testing.T) {
		base := newMLBoost()
		if _, err := base.WithParams(map[string]float64{"bogus": 1}); err == nil {
			t.Fatal("WithParams() error = nil, want error for unknown key")
		}
	})

	t.Run("trainWindow too small rejected", func(t *testing.T) {
		base := newMLBoost()
		if _, err := base.WithParams(map[string]float64{"trainWindow": 63}); err == nil {
			t.Fatal("WithParams() error = nil, want error for trainWindow <= feature floor")
		}
	})

	t.Run("non-integral trainWindow rejected", func(t *testing.T) {
		base := newMLBoost()
		if _, err := base.WithParams(map[string]float64{"trainWindow": 504.5}); err == nil {
			t.Fatal("WithParams() error = nil, want error for non-integral trainWindow")
		}
	})

	t.Run("negative deadband rejected", func(t *testing.T) {
		base := newMLBoost()
		if _, err := base.WithParams(map[string]float64{"deadband": -0.001}); err == nil {
			t.Fatal("WithParams() error = nil, want error for negative deadband")
		}
	})

	t.Run("receiver not mutated", func(t *testing.T) {
		base := newMLBoost()
		if _, err := base.WithParams(map[string]float64{"trainWindow": 756, "deadband": 0.0005}); err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		if base.trainWindow != 504 || base.deadband != 0.0 {
			t.Errorf("receiver mutated: trainWindow/deadband = %d/%v, want unchanged 504/0.0", base.trainWindow, base.deadband)
		}
	})

	t.Run("ParamSpace/Grid has the documented 4 combos", func(t *testing.T) {
		defs := newMLBoost().ParamSpace()
		combos := strategy.Grid(defs)
		if len(combos) != 4 {
			t.Fatalf("len(Grid(ParamSpace())) = %d, want 4", len(combos))
		}
	})
}

var _ strategy.Tunable = (*mlBoost)(nil)
var _ strategy.TargetWeighter = (*mlBoost)(nil)
