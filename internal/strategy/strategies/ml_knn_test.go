package strategies

import (
	"testing"

	"tradeforge/internal/strategy"
)

func TestMLKNN_Determinism(t *testing.T) {
	bars := scBarsFromCloses(mlRegimeCloses(8, 40))

	s1 := &mlKNN{m: 5, k: 10, libSize: 1000}
	s2 := &mlKNN{m: 5, k: 10, libSize: 1000}

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

func TestMLKNN_NoLookahead(t *testing.T) {
	bars := scBarsFromCloses(mlRegimeCloses(8, 40))

	n := 200
	sShort := &mlKNN{m: 5, k: 10, libSize: 1000}
	sLong := &mlKNN{m: 5, k: 10, libSize: 1000}

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

// TestMLKNN_Learnability reuses the alternating up/down regime series from
// ml_logit_test.go. Because each block is a pure arithmetic ramp, an m-day
// window drawn entirely from an up block always z-scores to the same
// canonical ascending shape (regardless of where in the block it starts),
// and a down block's windows z-score to the exact mirror-image descending
// shape — so a query pattern from deep inside an up block finds its nearest
// neighbors are (near-)exclusively other up-block windows, whose forward
// returns are, by construction, all positive.
func TestMLKNN_Learnability(t *testing.T) {
	closes := mlRegimeCloses(8, 40)
	bars := scBarsFromCloses(closes)

	t.Run("long: neighbors of an up-shaped pattern are reliably positive", func(t *testing.T) {
		// Bar 180 sits inside block 4 (indices 161..200, an up block).
		s := &mlKNN{m: 5, k: 10, libSize: 1000}
		history := bars[:181]
		ctx := strategy.NewContext(history, flatPortfolio())

		r1 := closes[180]/closes[179] - 1
		if r1 <= 0 {
			t.Fatalf("test setup invalid: bar 180's own return = %v, want > 0", r1)
		}

		weights := s.TargetWeights(ctx)
		if got, want := weights["SPY"], 1.0; got != want {
			t.Errorf("TargetWeights()[SPY] = %v, want %v (up-shaped neighbors should predict long)", got, want)
		}
	})

	t.Run("flat: neighbors of a down-shaped pattern are reliably negative", func(t *testing.T) {
		// Bar 220 sits inside block 5 (indices 201..240, a down block).
		s := &mlKNN{m: 5, k: 10, libSize: 1000}
		history := bars[:221]
		ctx := strategy.NewContext(history, flatPortfolio())

		r1 := closes[220]/closes[219] - 1
		if r1 >= 0 {
			t.Fatalf("test setup invalid: bar 220's own return = %v, want < 0", r1)
		}

		weights := s.TargetWeights(ctx)
		if len(weights) != 0 {
			t.Errorf("TargetWeights() = %+v, want empty map (down-shaped neighbors should not predict long)", weights)
		}
	})
}

func TestMLKNN_TargetWeights_PurityFromPortfolio(t *testing.T) {
	bars := scBarsFromCloses(mlRegimeCloses(8, 40))[:200]

	s1 := &mlKNN{m: 5, k: 10, libSize: 1000}
	s2 := &mlKNN{m: 5, k: 10, libSize: 1000}

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

func TestMLKNN_WithParams(t *testing.T) {
	t.Run("valid full override applied", func(t *testing.T) {
		base := newMLKNN()
		got, err := base.WithParams(map[string]float64{"m": 10, "k": 25})
		if err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		mk, ok := got.(*mlKNN)
		if !ok {
			t.Fatalf("WithParams() returned %T, want *mlKNN", got)
		}
		if mk.m != 10 || mk.k != 25 {
			t.Errorf("m/k = %d/%d, want 10/25", mk.m, mk.k)
		}
	})

	t.Run("partial map keeps defaults", func(t *testing.T) {
		base := newMLKNN()
		got, err := base.WithParams(map[string]float64{"k": 25})
		if err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		mk := got.(*mlKNN)
		if mk.k != 25 {
			t.Errorf("k = %d, want 25", mk.k)
		}
		if mk.m != 5 {
			t.Errorf("m = %d, want default 5", mk.m)
		}
	})

	t.Run("unknown key rejected", func(t *testing.T) {
		base := newMLKNN()
		if _, err := base.WithParams(map[string]float64{"bogus": 1}); err == nil {
			t.Fatal("WithParams() error = nil, want error for unknown key")
		}
	})

	t.Run("m < 2 rejected", func(t *testing.T) {
		base := newMLKNN()
		if _, err := base.WithParams(map[string]float64{"m": 1}); err == nil {
			t.Fatal("WithParams() error = nil, want error for m<2")
		}
	})

	t.Run("k < 1 rejected", func(t *testing.T) {
		base := newMLKNN()
		if _, err := base.WithParams(map[string]float64{"k": 0}); err == nil {
			t.Fatal("WithParams() error = nil, want error for k<1")
		}
	})

	t.Run("non-integral value rejected", func(t *testing.T) {
		base := newMLKNN()
		if _, err := base.WithParams(map[string]float64{"m": 5.5}); err == nil {
			t.Fatal("WithParams() error = nil, want error for non-integral m")
		}
	})

	t.Run("receiver not mutated", func(t *testing.T) {
		base := newMLKNN()
		if _, err := base.WithParams(map[string]float64{"m": 10, "k": 25}); err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		if base.m != 5 || base.k != 10 {
			t.Errorf("receiver mutated: m/k = %d/%d, want unchanged 5/10", base.m, base.k)
		}
	})

	t.Run("ParamSpace/Grid has the documented 4 combos", func(t *testing.T) {
		defs := newMLKNN().ParamSpace()
		combos := strategy.Grid(defs)
		if len(combos) != 4 {
			t.Fatalf("len(Grid(ParamSpace())) = %d, want 4", len(combos))
		}
	})
}

var _ strategy.Tunable = (*mlKNN)(nil)
var _ strategy.TargetWeighter = (*mlKNN)(nil)
