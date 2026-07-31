package strategies

import (
	"testing"

	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// mlRegimeCloses builds a deterministic synthetic close-price series made of
// alternating "up" and "down" blocks, shared by all three ML strategies'
// learnability tests (defined once here since ml_logit_test.go,
// ml_knn_test.go, and ml_boost_test.go are all in this package). Within an
// up block, daily returns form a RISING arithmetic sequence of positive
// values (every day is up, and — by construction — so is the next day: "up
// follows up", the momentum-autocorrelation pattern from the strategy
// spec); within a down block, returns form a FALLING arithmetic sequence of
// negative values (down follows down). The series alternates up, down,
// up, ..., starting with an up block (the spec's "seeded with one up day").
//
// Deviation from a flat +/-0.8% series, noted here rather than hidden: a
// flat run, after ml-knn's own per-window z-scoring, carries NO
// distinguishing shape between regimes (the constant level is exactly what
// z-scoring removes), which would make the k-NN shape-matching test
// untestable. Ramping the magnitude within each block gives every block a
// genuine, z-score-stable shape — any m-window drawn entirely from one
// arithmetic block reduces to the SAME canonical standardized vector
// regardless of where in the block it starts, and the up-block's canonical
// shape is the exact negative of the down-block's — so nearest-neighbor
// matching is clean and well-defined. The label relationship (tomorrow
// follows today's regime) is unchanged and remains 100% deterministic; only
// windows spanning a block boundary (a small minority) briefly violate it,
// same as real regime noise.
func mlRegimeCloses(numBlocks, blockLen int) []float64 {
	closes := make([]float64, 0, numBlocks*blockLen+1)
	closes = append(closes, 100.0)
	up := true
	for b := 0; b < numBlocks; b++ {
		for j := 0; j < blockLen; j++ {
			mag := 0.001 + float64(j)*0.0004
			ret := mag
			if !up {
				ret = -mag
			}
			last := closes[len(closes)-1]
			closes = append(closes, last*(1+ret))
		}
		up = !up
	}
	return closes
}

// runMLStrategy simulates the backtest engine's calling convention for a
// single-symbol strategy over bars: a flat, never-updated portfolio (these
// structural tests care about the SIGNAL SEQUENCE a strategy emits, not
// fill bookkeeping — the same simplification donchian_test.go and
// sma_cross_test.go make), OnBar called once per bar from WarmupBars()
// onward, using strategy.NewContext(bars[:i+1], port) each time — mirroring
// internal/backtest/engine.go's own no-lookahead calling convention (bar i
// sees history bars[:i+1], nothing later). Returns one signal slice per
// call (nil entries included) so callers can compare prefixes across two
// different bar-slice lengths fed to two independent instances.
func runMLStrategy(s strategy.Strategy, bars []domain.Bar) [][]domain.Signal {
	port := flatPortfolio()
	warmup := s.WarmupBars()
	out := make([][]domain.Signal, 0, len(bars))
	for i := warmup; i < len(bars); i++ {
		ctx := strategy.NewContext(bars[:i+1], port)
		out = append(out, s.OnBar(ctx, bars[i]))
	}
	return out
}

// signalsEqual reports whether two signal slices carry the same symbols and
// target weights (order-independent would be overkill here — OnBar always
// emits at most one signal for these single-symbol strategies — so a simple
// length + element comparison suffices).
func signalsEqual(a, b []domain.Signal) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Symbol != b[i].Symbol || a[i].TargetWeight != b[i].TargetWeight {
			return false
		}
	}
	return true
}

func TestMLLogit_Determinism(t *testing.T) {
	bars := scBarsFromCloses(mlRegimeCloses(8, 40))

	s1 := &mlLogit{trainWindow: 100, enterThresh: 0.55, refitEvery: 21}
	s2 := &mlLogit{trainWindow: 100, enterThresh: 0.55, refitEvery: 21}

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

func TestMLLogit_NoLookahead(t *testing.T) {
	bars := scBarsFromCloses(mlRegimeCloses(8, 40))

	n := 200
	sShort := &mlLogit{trainWindow: 100, enterThresh: 0.55, refitEvery: 21}
	sLong := &mlLogit{trainWindow: 100, enterThresh: 0.55, refitEvery: 21}

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

// TestMLLogit_Learnability builds the alternating up/down regime series and
// checks that, once past warmup, the model goes long on a bar deep inside
// an up block (today's own return positive — "the bar after an up day" in
// the sense that r1, today's return, feeds directly into the model that
// predicts tomorrow) and stays flat on a bar deep inside a down block.
// TestMLLogit_FreshInstanceMatchesLongLived pins the backtest/paper
// equivalence property: a backtest drives ONE long-lived instance bar by
// bar, while the paper loop constructs a FRESH instance each session and
// calls it once on full history. The governing model must be a pure
// function of history — never of process lifetime — so at every bar a
// fresh one-shot instance must emit exactly what the long-lived instance
// does. Regression for the 2026-07-15 review finding: the old "refit when
// no cached model exists yet" rule made a fresh process refit daily while
// the backtest refit on cadence, so the strategy that traded in paper was
// not the strategy the backtest validated.
func TestMLLogit_FreshInstanceMatchesLongLived(t *testing.T) {
	bars := scBarsFromCloses(mlRegimeCloses(10, 40))
	port := flatPortfolio()

	long := &mlLogit{trainWindow: 100, enterThresh: 0.55, refitEvery: 21}
	sawSignal := false
	for i := long.WarmupBars(); i < len(bars); i++ {
		ctx := strategy.NewContext(bars[:i+1], port)
		longSigs := long.OnBar(ctx, bars[i])

		fresh := &mlLogit{trainWindow: 100, enterThresh: 0.55, refitEvery: 21}
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

// TestMLLogitFeatures_ZeroCloseGuard: one 0.0 close inside a momentum
// window must invalidate the feature vector (ok=false) rather than sending
// an Inf through standardization and NaN-ing the fitted model.
func TestMLLogitFeatures_ZeroCloseGuard(t *testing.T) {
	closes := make([]float64, 120)
	for i := range closes {
		closes[i] = 100
	}
	closes[95] = 0 // i-5 for i=100
	if _, ok := mlLogitFeatures(closes, 100); ok {
		t.Fatal("mlLogitFeatures ok = true with a zero close in the momentum window, want false")
	}
}

func TestMLLogit_Learnability(t *testing.T) {
	closes := mlRegimeCloses(8, 40)
	bars := scBarsFromCloses(closes)

	t.Run("long after an up day", func(t *testing.T) {
		// Bar 180 sits inside block 4 (indices 161..200, an up block):
		// today's own return is positive.
		s := &mlLogit{trainWindow: 100, enterThresh: 0.55, refitEvery: 21}
		history := bars[:181]
		ctx := strategy.NewContext(history, flatPortfolio())

		r1 := closes[180]/closes[179] - 1
		if r1 <= 0 {
			t.Fatalf("test setup invalid: bar 180's own return = %v, want > 0", r1)
		}

		weights := s.TargetWeights(ctx)
		if got, want := weights["SPY"], 1.0; got != want {
			t.Errorf("TargetWeights()[SPY] = %v, want %v (up-day feature should predict long)", got, want)
		}
	})

	t.Run("flat after a down day", func(t *testing.T) {
		// Bar 220 sits inside block 5 (indices 201..240, a down block):
		// today's own return is negative.
		s := &mlLogit{trainWindow: 100, enterThresh: 0.55, refitEvery: 21}
		history := bars[:221]
		ctx := strategy.NewContext(history, flatPortfolio())

		r1 := closes[220]/closes[219] - 1
		if r1 >= 0 {
			t.Fatalf("test setup invalid: bar 220's own return = %v, want < 0", r1)
		}

		weights := s.TargetWeights(ctx)
		if len(weights) != 0 {
			t.Errorf("TargetWeights() = %+v, want empty map (down-day feature should not predict long)", weights)
		}
	})
}

func TestMLLogit_TargetWeights_PurityFromPortfolio(t *testing.T) {
	bars := scBarsFromCloses(mlRegimeCloses(8, 40))[:200]

	s1 := &mlLogit{trainWindow: 100, enterThresh: 0.55, refitEvery: 21}
	s2 := &mlLogit{trainWindow: 100, enterThresh: 0.55, refitEvery: 21}

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

func TestMLLogit_WithParams(t *testing.T) {
	t.Run("valid full override applied", func(t *testing.T) {
		base := newMLLogit()
		got, err := base.WithParams(map[string]float64{"trainWindow": 504, "enterThresh": 0.57})
		if err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		ml, ok := got.(*mlLogit)
		if !ok {
			t.Fatalf("WithParams() returned %T, want *mlLogit", got)
		}
		if ml.trainWindow != 504 || ml.enterThresh != 0.57 {
			t.Errorf("trainWindow/enterThresh = %d/%v, want 504/0.57", ml.trainWindow, ml.enterThresh)
		}
	})

	t.Run("partial map keeps defaults", func(t *testing.T) {
		base := newMLLogit()
		got, err := base.WithParams(map[string]float64{"enterThresh": 0.53})
		if err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		ml := got.(*mlLogit)
		if ml.enterThresh != 0.53 {
			t.Errorf("enterThresh = %v, want 0.53", ml.enterThresh)
		}
		if ml.trainWindow != 252 {
			t.Errorf("trainWindow = %d, want default 252", ml.trainWindow)
		}
	})

	t.Run("unknown key rejected", func(t *testing.T) {
		base := newMLLogit()
		if _, err := base.WithParams(map[string]float64{"bogus": 1}); err == nil {
			t.Fatal("WithParams() error = nil, want error for unknown key")
		}
	})

	t.Run("trainWindow too small rejected", func(t *testing.T) {
		base := newMLLogit()
		if _, err := base.WithParams(map[string]float64{"trainWindow": 63}); err == nil {
			t.Fatal("WithParams() error = nil, want error for trainWindow <= feature floor")
		}
	})

	t.Run("non-integral trainWindow rejected", func(t *testing.T) {
		base := newMLLogit()
		if _, err := base.WithParams(map[string]float64{"trainWindow": 252.5}); err == nil {
			t.Fatal("WithParams() error = nil, want error for non-integral trainWindow")
		}
	})

	t.Run("enterThresh out of range rejected", func(t *testing.T) {
		base := newMLLogit()
		if _, err := base.WithParams(map[string]float64{"enterThresh": 0.5}); err == nil {
			t.Fatal("WithParams() error = nil, want error for enterThresh<=0.5")
		}
		if _, err := base.WithParams(map[string]float64{"enterThresh": 1.0}); err == nil {
			t.Fatal("WithParams() error = nil, want error for enterThresh>=1")
		}
	})

	t.Run("receiver not mutated", func(t *testing.T) {
		base := newMLLogit()
		if _, err := base.WithParams(map[string]float64{"trainWindow": 504, "enterThresh": 0.57}); err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		if base.trainWindow != 252 || base.enterThresh != 0.55 {
			t.Errorf("receiver mutated: trainWindow/enterThresh = %d/%v, want unchanged 252/0.55", base.trainWindow, base.enterThresh)
		}
	})

	t.Run("ParamSpace/Grid has the documented 6 combos", func(t *testing.T) {
		defs := newMLLogit().ParamSpace()
		combos := strategy.Grid(defs)
		if len(combos) != 6 {
			t.Fatalf("len(Grid(ParamSpace())) = %d, want 6", len(combos))
		}
	})
}

var _ strategy.Tunable = (*mlLogit)(nil)
var _ strategy.TargetWeighter = (*mlLogit)(nil)
