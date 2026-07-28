package strategies

import (
	"time"

	"testing"

	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// vtDay returns a deterministic UTC date offset from a fixed epoch.
func vtDay(n int) time.Time {
	return time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC).AddDate(0, 0, n)
}

// vtBarsFromCloses builds bars for symbol "V" from a slice of close prices,
// oldest first. Open/High/Low all equal Close for simplicity.
func vtBarsFromCloses(closes []float64) []domain.Bar {
	bars := make([]domain.Bar, len(closes))
	for i, c := range closes {
		bars[i] = domain.Bar{
			Symbol: "V",
			Time:   vtDay(i),
			Open:   c, High: c, Low: c, Close: c,
			Volume: 1000,
		}
	}
	return bars
}

// vtLowVolSeries builds a rising series with small daily steps (low
// realized vol): alternating +0.06%/+0.04% (average +0.05% per day) so the
// series has a small but nonzero return dispersion — a perfectly constant
// daily return would have EXACTLY zero sample stdev, which is the
// degenerate case tested separately.
func vtLowVolSeries(n int, start float64) []float64 {
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

// vtHighVolSeries builds an alternating +/-3% series (high realized vol).
func vtHighVolSeries(n int, start float64) []float64 {
	closes := make([]float64, n)
	c := start
	for i := range closes {
		closes[i] = c
		if i%2 == 0 {
			c *= 1.03
		} else {
			c *= 0.97
		}
	}
	return closes
}

func vtFlatPortfolio() *domain.Portfolio {
	return domain.NewPortfolio(100000)
}

func TestVolTarget_RisingLowVolSeries_WeightNearCapAndSignalWhenFlat(t *testing.T) {
	s := newVolTarget()
	s.lookback = 20
	s.checkEvery = 21
	// vtFlatPortfolio() below is flat, so the flat-book bootstrap opens the
	// gate regardless of cadence.

	closes := vtLowVolSeries(25, 100)
	history := vtBarsFromCloses(closes)
	ctx := strategy.NewContext(history, vtFlatPortfolio())

	sigs := s.OnBar(ctx, history[len(history)-1])
	if len(sigs) != 1 {
		t.Fatalf("OnBar() signals = %+v, want exactly 1 signal (flat -> desired weight)", sigs)
	}
	if sigs[0].Symbol != "V" {
		t.Errorf("Symbol = %q, want V", sigs[0].Symbol)
	}
	// Low vol -> targetVol/realizedVol should clamp at or near the 1.0 cap.
	if sigs[0].TargetWeight < 0.9 {
		t.Errorf("TargetWeight = %v, want near the 1.0 cap for a low-vol series", sigs[0].TargetWeight)
	}
}

func TestVolTarget_HighVolSeries_ShrinksWeight(t *testing.T) {
	s := newVolTarget()
	s.lookback = 20
	s.checkEvery = 21
	// vtFlatPortfolio() below is flat, so the flat-book bootstrap opens the
	// gate regardless of cadence.

	closes := vtHighVolSeries(25, 100)
	history := vtBarsFromCloses(closes)
	ctx := strategy.NewContext(history, vtFlatPortfolio())

	sigs := s.OnBar(ctx, history[len(history)-1])
	if len(sigs) != 1 {
		t.Fatalf("OnBar() signals = %+v, want exactly 1 signal", sigs)
	}
	// A ~3% daily swing annualizes to roughly 0.03*sqrt(252) ~= 0.48 realized
	// vol; targetVol 0.10 / 0.48 ~= 0.21, well under the cap.
	if sigs[0].TargetWeight >= 0.5 {
		t.Errorf("TargetWeight = %v, want well under 1.0 for a high-vol series", sigs[0].TargetWeight)
	}
	if sigs[0].TargetWeight <= 0 {
		t.Errorf("TargetWeight = %v, want > 0", sigs[0].TargetWeight)
	}
}

func TestVolTarget_NoSignalWithinBand(t *testing.T) {
	s := newVolTarget()
	s.lookback = 20
	s.checkEvery = 21
	// closes below has exactly 21 bars, matching checkEvery exactly
	// (histLen % checkEvery == 0), arming the cadence gate explicitly since
	// the portfolio built below holds a position in V (not flat).

	closes := vtLowVolSeries(21, 100)
	history := vtBarsFromCloses(closes)

	w, ok := s.desiredWeight(strategy.Closes(history))
	if !ok {
		t.Fatalf("desiredWeight() ok = false, want true")
	}

	// Build a portfolio already holding a position whose derived weight
	// equals the desired weight w exactly, so |w - wCurrent| == 0 <= 0.05.
	lastClose := history[len(history)-1].Close
	equity := 100000.0
	qty := int64(w * equity / lastClose)
	p := domain.NewPortfolio(equity - float64(qty)*lastClose)
	p.Positions["V"] = domain.Position{Symbol: "V", Qty: qty, AvgPrice: lastClose}

	ctx := strategy.NewContext(history, p)
	sigs := s.OnBar(ctx, history[len(history)-1])
	if len(sigs) != 0 {
		t.Fatalf("OnBar() signals = %+v, want none (within the 0.05 band)", sigs)
	}
}

func TestVolTarget_CadenceRespected(t *testing.T) {
	s := newVolTarget()
	s.lookback = 20
	s.checkEvery = 21
	// 25 bars: 25 % 21 == 4, not a cadence multiple. The portfolio below
	// HOLDS a position in V, so the flat-book bootstrap does not fire
	// either — this is the "held and not due" case, which must stay nil.

	closes := vtLowVolSeries(25, 100)
	history := vtBarsFromCloses(closes)
	p := domain.NewPortfolio(100000)
	p.Positions["V"] = domain.Position{Symbol: "V", Qty: 100, AvgPrice: history[len(history)-1].Close}
	ctx := strategy.NewContext(history, p)

	sigs := s.OnBar(ctx, history[len(history)-1])
	if sigs != nil {
		t.Fatalf("OnBar() signals = %+v, want nil off the checkEvery cadence while holding a position", sigs)
	}
}

func TestVolTarget_DegenerateFlatSeries_ZeroWeight(t *testing.T) {
	s := newVolTarget()
	s.lookback = 20
	s.checkEvery = 21

	closes := make([]float64, 25)
	for i := range closes {
		closes[i] = 100
	}
	history := vtBarsFromCloses(closes)

	w, ok := s.desiredWeight(strategy.Closes(history))
	if !ok {
		t.Fatalf("desiredWeight() ok = false, want true")
	}
	if w != 0 {
		t.Errorf("desiredWeight() = %v, want 0 for a zero-vol degenerate series", w)
	}
}

func TestVolTarget_WarmupBars(t *testing.T) {
	s := newVolTarget()
	if got, want := s.WarmupBars(), s.lookback+1; got != want {
		t.Errorf("WarmupBars() = %d, want %d", got, want)
	}
}

func TestVolTarget_WithParams(t *testing.T) {
	base := newVolTarget()

	t.Run("valid override applied", func(t *testing.T) {
		got, err := base.WithParams(map[string]float64{"targetVol": 0.08, "lookback": 60})
		if err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		vt := got.(*volTarget)
		if vt.targetVol != 0.08 {
			t.Errorf("targetVol = %v, want 0.08", vt.targetVol)
		}
		if vt.lookback != 60 {
			t.Errorf("lookback = %d, want 60", vt.lookback)
		}
		if vt.checkEvery != 21 {
			t.Errorf("checkEvery = %d, want 21 (fixed, not tunable)", vt.checkEvery)
		}
	})

	t.Run("receiver not mutated", func(t *testing.T) {
		_, err := base.WithParams(map[string]float64{"targetVol": 0.08})
		if err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		if base.targetVol != 0.10 {
			t.Errorf("receiver mutated: targetVol = %v, want unchanged 0.10", base.targetVol)
		}
	})

	t.Run("partial map keeps defaults", func(t *testing.T) {
		got, err := base.WithParams(map[string]float64{"targetVol": 0.12})
		if err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		vt := got.(*volTarget)
		if vt.lookback != 20 {
			t.Errorf("lookback = %d, want default 20", vt.lookback)
		}
	})

	t.Run("unknown key rejected", func(t *testing.T) {
		if _, err := base.WithParams(map[string]float64{"checkEvery": 10}); err == nil {
			t.Fatal("expected an error for the unknown key checkEvery (fixed, not tunable)")
		}
	})

	t.Run("targetVol <= 0 rejected", func(t *testing.T) {
		if _, err := base.WithParams(map[string]float64{"targetVol": 0}); err == nil {
			t.Fatal("expected an error for targetVol == 0")
		}
	})

	t.Run("targetVol >= 1 rejected", func(t *testing.T) {
		if _, err := base.WithParams(map[string]float64{"targetVol": 1}); err == nil {
			t.Fatal("expected an error for targetVol == 1")
		}
	})

	t.Run("non-integral lookback rejected", func(t *testing.T) {
		if _, err := base.WithParams(map[string]float64{"lookback": 20.5}); err == nil {
			t.Fatal("expected an error for non-integral lookback")
		}
	})

	t.Run("lookback below minimum rejected", func(t *testing.T) {
		if _, err := base.WithParams(map[string]float64{"lookback": 4}); err == nil {
			t.Fatal("expected an error for lookback < 5")
		}
	})

	t.Run("unknown key alone rejected", func(t *testing.T) {
		if _, err := base.WithParams(map[string]float64{"bogus": 1}); err == nil {
			t.Fatal("expected an error for an unknown key")
		}
	})
}

// TestVolTarget_PaperRestart_FreshInstanceFlatBootstraps reproduces the
// paper loop's call shape: a brand-new instance (as constructed fresh every
// session), called exactly once, with a bar count that is NOT a checkEvery
// multiple. Before the derive-from-data fix this returned nil forever,
// because barsSeen always started (and stayed) at 1 in a fresh-per-session
// process — the strategy could never trade in paper. The flat-book
// bootstrap must fire instead and produce a sizing signal.
func TestVolTarget_PaperRestart_FreshInstanceFlatBootstraps(t *testing.T) {
	s := newVolTarget() // defaults: lookback=20, checkEvery=21

	closes := vtLowVolSeries(25, 100) // 25 bars; 25 % 21 == 4, not a cadence multiple
	history := vtBarsFromCloses(closes)
	ctx := strategy.NewContext(history, vtFlatPortfolio())

	sigs := s.OnBar(ctx, history[len(history)-1])
	if len(sigs) != 1 {
		t.Fatalf("OnBar() signals = %+v, want exactly 1 signal on the very first (flat, off-cadence) paper call", sigs)
	}
}

// TestVolTarget_PaperRestart_HeldPositionNotDue_Nil is the companion to the
// bootstrap test above: same fresh-instance, single-call paper-restart
// shape, but the portfolio already holds V, so the flat-book bootstrap does
// not fire, and the bar count is deliberately not a checkEvery multiple.
// The correct behavior is nil.
func TestVolTarget_PaperRestart_HeldPositionNotDue_Nil(t *testing.T) {
	s := newVolTarget()

	closes := vtLowVolSeries(25, 100)
	history := vtBarsFromCloses(closes)
	p := domain.NewPortfolio(100000)
	p.Positions["V"] = domain.Position{Symbol: "V", Qty: 100, AvgPrice: history[len(history)-1].Close}
	ctx := strategy.NewContext(history, p)

	sigs := s.OnBar(ctx, history[len(history)-1])
	if sigs != nil {
		t.Fatalf("OnBar() signals = %+v, want nil: held position + off-cadence tick", sigs)
	}
}

var _ strategy.Tunable = (*volTarget)(nil)
