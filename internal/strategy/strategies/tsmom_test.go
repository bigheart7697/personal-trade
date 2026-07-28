package strategies

import (
	"testing"
	"time"

	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// tsDay returns a deterministic UTC date offset from a fixed epoch.
func tsDay(n int) time.Time {
	return time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC).AddDate(0, 0, n)
}

// tsBarsFromCloses builds bars for symbol "M" from a slice of close prices,
// oldest first. Open/High/Low all equal Close for simplicity — tsmom's
// signal is close-only.
func tsBarsFromCloses(closes []float64) []domain.Bar {
	bars := make([]domain.Bar, len(closes))
	for i, c := range closes {
		bars[i] = domain.Bar{
			Symbol: "M",
			Time:   tsDay(i),
			Open:   c, High: c, Low: c, Close: c,
			Volume: 1000,
		}
	}
	return bars
}

// tsRisingLowVolSeries builds a rising series with small alternating daily
// steps (+0.06%/+0.04%) — comfortably positive trailing momentum with very
// low realized volatility, so targetVol/realizedVol should clamp at the 1.0
// cap. Same construction as vol-target's own low-vol series (vtLowVolSeries)
// for the same reason: a perfectly constant daily return would have exactly
// zero sample stdev, the degenerate case handled separately.
func tsRisingLowVolSeries(n int, start float64) []float64 {
	closes := make([]float64, n)
	c := start
	for i := range closes {
		closes[i] = c
		if i%2 == 0 {
			c *= 1.0006
		} else {
			c *= 1.0004
		}
	}
	return closes
}

// tsFallingSeries builds a steadily falling series so trailing momentum
// measured over it is unambiguously negative.
func tsFallingSeries(n int, start float64) []float64 {
	closes := make([]float64, n)
	c := start
	for i := range closes {
		closes[i] = c
		c *= 0.995
	}
	return closes
}

func TestTSMOM_Determinism(t *testing.T) {
	closes := tsRisingLowVolSeries(200, 100)
	bars := tsBarsFromCloses(closes)

	s1 := &tsmom{lookback: 60, targetVol: 0.10}
	s2 := &tsmom{lookback: 60, targetVol: 0.10}

	sigs1 := runMLStrategy(s1, bars)
	sigs2 := runMLStrategy(s2, bars)

	if len(sigs1) != len(sigs2) {
		t.Fatalf("signal sequence lengths differ: %d vs %d", len(sigs1), len(sigs2))
	}
	for i := range sigs1 {
		if !signalsEqual(sigs1[i], sigs2[i]) {
			t.Fatalf("signal at index %d differs between two fresh instances fed identical bars: %+v vs %+v", i, sigs1[i], sigs2[i])
		}
	}
}

func TestTSMOM_NoLookahead(t *testing.T) {
	closes := tsRisingLowVolSeries(250, 100)
	bars := tsBarsFromCloses(closes)

	n := 200
	sShort := &tsmom{lookback: 60, targetVol: 0.10}
	sLong := &tsmom{lookback: 60, targetVol: 0.10}

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

// TestTSMOM_TargetWeights_RisingYear_ClampsAtCap builds a history whose
// length (147 = 21*7) is itself an exact cadence tick, so TargetWeights'
// cadence lock introduces no truncation ambiguity — the answer must equal
// momentumAndWeight computed directly on the full history.
func TestTSMOM_TargetWeights_RisingYear_ClampsAtCap(t *testing.T) {
	s := &tsmom{lookback: 60, targetVol: 0.10}

	closes := tsRisingLowVolSeries(147, 100)
	history := tsBarsFromCloses(closes)
	ctx := strategy.NewContext(history, flatPortfolio())

	weights := s.TargetWeights(ctx)
	if got, want := weights["M"], 1.0; got != want {
		t.Errorf("TargetWeights()[M] = %v, want exactly %v (targetVol/realizedVol should clamp at the 1.0 cap for a low-vol rising series)", got, want)
	}
}

func TestTSMOM_TargetWeights_FallingYear_Flat(t *testing.T) {
	s := &tsmom{lookback: 60, targetVol: 0.10}

	closes := tsFallingSeries(147, 100)
	history := tsBarsFromCloses(closes)
	ctx := strategy.NewContext(history, flatPortfolio())

	weights := s.TargetWeights(ctx)
	if len(weights) != 0 {
		t.Errorf("TargetWeights() = %+v, want empty map for a falling trailing-year momentum", weights)
	}
}

// TestTSMOM_TargetWeights_BootstrapWhenFlatAtLastTick exercises the "book is
// flat while mom > 0" bootstrap: a series that is flat through bar 104 and
// then ramps sharply upward. With lookback=60, the last cadence tick <= 130
// bars is 126, and momentum measured AS OF bar 125 (indices 104 and 65, both
// still in the flat era) is exactly zero — the cadence-locked answer alone
// would stay flat. Momentum measured as of the CURRENT bar 129 (indices 108
// and 69) has already caught the ramp and is positive. TargetWeights must
// use today's value instead of waiting for the next monthly tick.
func TestTSMOM_TargetWeights_BootstrapWhenFlatAtLastTick(t *testing.T) {
	s := &tsmom{lookback: 60, targetVol: 0.10}

	const n = 130
	closes := make([]float64, n)
	for i := range closes {
		if i <= 104 {
			closes[i] = 100
		} else {
			closes[i] = 100 + float64(i-104)*5
		}
	}
	history := tsBarsFromCloses(closes)
	ctx := strategy.NewContext(history, flatPortfolio())

	// Sanity-check the test setup directly against the pure helper.
	tickWeight, ok := s.momentumAndWeight(history[:126])
	if !ok || tickWeight != 0 {
		t.Fatalf("test setup invalid: momentumAndWeight(history[:126]) = (%v,%v), want (0,true)", tickWeight, ok)
	}
	nowWeight, ok := s.momentumAndWeight(history)
	if !ok || nowWeight <= 0 {
		t.Fatalf("test setup invalid: momentumAndWeight(history) = (%v,%v), want a positive weight", nowWeight, ok)
	}

	weights := s.TargetWeights(ctx)
	if got, want := weights["M"], nowWeight; got != want {
		t.Errorf("TargetWeights()[M] = %v, want %v (bootstrap should use today's momentum when the last cadence tick was flat)", got, want)
	}
}

// TestTSMOM_OnBar_CadenceRespected checks OnBar's own gate (distinct from
// TargetWeights' cadence lock): with a non-flat portfolio and a history
// length that is NOT a cadence multiple, OnBar must not signal at all, no
// matter how far the (risk-capped) current weight has drifted from the raw
// target — see OnBar's doc comment for why this gate exists.
func TestTSMOM_OnBar_CadenceRespected(t *testing.T) {
	s := &tsmom{lookback: 60, targetVol: 0.10}

	closes := tsRisingLowVolSeries(130, 100) // 130 % 21 == 4, not a cadence multiple
	history := tsBarsFromCloses(closes)
	p := domain.NewPortfolio(100000)
	p.Positions["M"] = domain.Position{Symbol: "M", Qty: 1, AvgPrice: history[len(history)-1].Close}
	ctx := strategy.NewContext(history, p)

	sig := s.OnBar(ctx, history[len(history)-1])
	if sig != nil {
		t.Fatalf("OnBar() signals = %+v, want nil off the cadence while holding a position", sig)
	}
}

// TestTSMOM_OnBar_FlatBootstrapFiresOffCadence is the companion case: a
// flat book, off-cadence, with a positive raw target — the flat-book
// bootstrap must still let OnBar signal immediately rather than waiting for
// the next monthly tick.
func TestTSMOM_OnBar_FlatBootstrapFiresOffCadence(t *testing.T) {
	s := &tsmom{lookback: 60, targetVol: 0.10}

	closes := tsRisingLowVolSeries(130, 100) // 130 % 21 == 4, not a cadence multiple
	history := tsBarsFromCloses(closes)
	ctx := strategy.NewContext(history, flatPortfolio())

	sig := s.OnBar(ctx, history[len(history)-1])
	if len(sig) != 1 {
		t.Fatalf("OnBar() signals = %+v, want exactly 1 signal (flat bootstrap should fire even off-cadence)", sig)
	}
}

func TestTSMOM_TargetWeights_InsufficientHistory(t *testing.T) {
	s := newTSMOM()
	closes := tsRisingLowVolSeries(10, 100)
	history := tsBarsFromCloses(closes)
	ctx := strategy.NewContext(history, flatPortfolio())

	weights := s.TargetWeights(ctx)
	if len(weights) != 0 {
		t.Errorf("TargetWeights() = %+v, want empty map when history is shorter than lookback+1", weights)
	}
}

func TestTSMOM_TargetWeights_PurityFromPortfolio(t *testing.T) {
	closes := tsRisingLowVolSeries(147, 100)
	history := tsBarsFromCloses(closes)

	s1 := &tsmom{lookback: 60, targetVol: 0.10}
	s2 := &tsmom{lookback: 60, targetVol: 0.10}

	ctxFlat := strategy.NewContext(history, flatPortfolio())
	ctxLong := strategy.NewContext(history, longPortfolio(50, 100))

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

func TestTSMOM_WarmupBars(t *testing.T) {
	s := newTSMOM()
	if got, want := s.WarmupBars(), s.lookback+64; got != want {
		t.Errorf("WarmupBars() = %d, want %d", got, want)
	}
}

func TestTSMOM_WithParams(t *testing.T) {
	base := newTSMOM()

	t.Run("valid full override applied", func(t *testing.T) {
		got, err := base.WithParams(map[string]float64{"lookback": 126, "targetVol": 0.15})
		if err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		ts, ok := got.(*tsmom)
		if !ok {
			t.Fatalf("WithParams() returned %T, want *tsmom", got)
		}
		if ts.lookback != 126 || ts.targetVol != 0.15 {
			t.Errorf("lookback/targetVol = %d/%v, want 126/0.15", ts.lookback, ts.targetVol)
		}
	})

	t.Run("partial map keeps defaults", func(t *testing.T) {
		got, err := base.WithParams(map[string]float64{"targetVol": 0.15})
		if err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		ts := got.(*tsmom)
		if ts.targetVol != 0.15 {
			t.Errorf("targetVol = %v, want 0.15", ts.targetVol)
		}
		if ts.lookback != 252 {
			t.Errorf("lookback = %d, want default 252", ts.lookback)
		}
	})

	t.Run("unknown key rejected", func(t *testing.T) {
		if _, err := base.WithParams(map[string]float64{"bogus": 1}); err == nil {
			t.Fatal("WithParams() error = nil, want error for unknown key")
		}
	})

	t.Run("non-integral lookback rejected", func(t *testing.T) {
		if _, err := base.WithParams(map[string]float64{"lookback": 126.5}); err == nil {
			t.Fatal("WithParams() error = nil, want error for non-integral lookback")
		}
	})

	t.Run("lookback <= fixed skip rejected", func(t *testing.T) {
		if _, err := base.WithParams(map[string]float64{"lookback": 21}); err == nil {
			t.Fatal("WithParams() error = nil, want error for lookback<=skip")
		}
	})

	t.Run("targetVol <= 0 rejected", func(t *testing.T) {
		if _, err := base.WithParams(map[string]float64{"targetVol": 0}); err == nil {
			t.Fatal("WithParams() error = nil, want error for targetVol<=0")
		}
	})

	t.Run("targetVol >= 1 rejected", func(t *testing.T) {
		if _, err := base.WithParams(map[string]float64{"targetVol": 1}); err == nil {
			t.Fatal("WithParams() error = nil, want error for targetVol>=1")
		}
	})

	t.Run("receiver not mutated", func(t *testing.T) {
		if _, err := base.WithParams(map[string]float64{"lookback": 126, "targetVol": 0.15}); err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		if base.lookback != 252 || base.targetVol != 0.10 {
			t.Errorf("receiver mutated: lookback/targetVol = %d/%v, want unchanged 252/0.10", base.lookback, base.targetVol)
		}
	})

	t.Run("ParamSpace/Grid has the documented 4 combos", func(t *testing.T) {
		defs := newTSMOM().ParamSpace()
		combos := strategy.Grid(defs)
		if len(combos) != 4 {
			t.Fatalf("len(Grid(ParamSpace())) = %d, want 4", len(combos))
		}
	})
}

var _ strategy.Tunable = (*tsmom)(nil)
var _ strategy.TargetWeighter = (*tsmom)(nil)
