package strategies

import (
	"testing"
	"time"

	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// bsBar builds a boll-snapback test bar for symbol "T" on day `day` with the
// given close (open/high/low are set equal to close; these tests only need
// closes for SMA/StdDev/regime checks).
func bsBar(day int, c float64) domain.Bar {
	return domain.Bar{
		Symbol: "T",
		Time:   time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC).AddDate(0, 0, day),
		Open:   c, High: c, Low: c, Close: c,
		Volume: 1000,
	}
}

// uptrendHistory builds n bars of a steadily rising close series starting
// at start and incrementing by step each bar, which keeps close comfortably
// above its own SMA200 once warmed up (a clean uptrend regime).
func uptrendHistory(n int, start, step float64) []domain.Bar {
	bars := make([]domain.Bar, n)
	for i := 0; i < n; i++ {
		bars[i] = bsBar(i, start+float64(i)*step)
	}
	return bars
}

func TestBollSnapback_OnBar(t *testing.T) {
	s, err := newBollSnapback().WithParams(map[string]float64{"period": 20, "k": 2.0, "timeStop": 10})
	if err != nil {
		t.Fatalf("WithParams() error = %v", err)
	}

	t.Run("entry fires below the lower band in an uptrend", func(t *testing.T) {
		bsInst := s.(*bollSnapback)
		fresh := &bollSnapback{smaPeriod: bsInst.smaPeriod, period: bsInst.period, k: bsInst.k, timeStop: bsInst.timeStop, entryWeight: bsInst.entryWeight}

		// 200+ rising bars establish an uptrend (close > SMA200), then a
		// sharp one-bar dip pushes today's close below SMA(20) - 2*stdev
		// while still trading above SMA200.
		history := uptrendHistory(220, 100, 0.5)
		last := history[len(history)-1]
		dip := bsBar(len(history), last.Close-20) // sharp single-bar drop
		history = append(history, dip)

		ctx := strategy.NewContext(history, flatPortfolio())
		sig := fresh.OnBar(ctx, history[len(history)-1])
		if len(sig) != 1 {
			t.Fatalf("OnBar() signals = %d, want 1", len(sig))
		}
		if sig[0].TargetWeight != 0.20 {
			t.Errorf("TargetWeight = %v, want 0.20", sig[0].TargetWeight)
		}
	})

	t.Run("no entry when below SMA200 (regime filter fails)", func(t *testing.T) {
		bsInst := s.(*bollSnapback)
		fresh := &bollSnapback{smaPeriod: bsInst.smaPeriod, period: bsInst.period, k: bsInst.k, timeStop: bsInst.timeStop, entryWeight: bsInst.entryWeight}

		// A downtrend: close is below its own SMA200, so even a dip below
		// the lower band must not trigger an entry (regime filter fails).
		history := uptrendHistory(220, 300, -0.5)
		last := history[len(history)-1]
		dip := bsBar(len(history), last.Close-20)
		history = append(history, dip)

		ctx := strategy.NewContext(history, flatPortfolio())
		sig := fresh.OnBar(ctx, history[len(history)-1])
		if len(sig) != 0 {
			t.Fatalf("OnBar() signals = %d, want 0 (regime filter should block entry)", len(sig))
		}
	})

	t.Run("no entry within the band", func(t *testing.T) {
		bsInst := s.(*bollSnapback)
		fresh := &bollSnapback{smaPeriod: bsInst.smaPeriod, period: bsInst.period, k: bsInst.k, timeStop: bsInst.timeStop, entryWeight: bsInst.entryWeight}

		// Flat-ish uptrend, no sharp dip: today's close sits inside the
		// band (close > SMA200, but not below SMA20 - 2*stdev).
		history := uptrendHistory(221, 100, 0.5)

		ctx := strategy.NewContext(history, flatPortfolio())
		sig := fresh.OnBar(ctx, history[len(history)-1])
		if len(sig) != 0 {
			t.Fatalf("OnBar() signals = %d, want 0 (inside the band)", len(sig))
		}
	})

	t.Run("exit on mean touch (independent of age)", func(t *testing.T) {
		bsInst := s.(*bollSnapback)
		fresh := &bollSnapback{smaPeriod: bsInst.smaPeriod, period: bsInst.period, k: bsInst.k, timeStop: bsInst.timeStop, entryWeight: bsInst.entryWeight}

		// Build history where SMA(20) is comfortably below today's close,
		// simulating a snap-back to (or above) the mean while long.
		history := uptrendHistory(220, 100, 0.5)
		sma, ok := strategy.SMA(strategy.Closes(history), fresh.period)
		if !ok {
			t.Fatalf("test setup: SMA() ok = false")
		}
		snapBar := bsBar(len(history), sma+1) // close snapped back above SMA
		history = append(history, snapBar)

		// Age 3 — mid-hold, well under the time stop; the mean-touch exit
		// must fire regardless.
		ctx := strategy.NewContext(history, longPortfolio(10, 90)).WithPositionAge(map[string]int{"T": 3})
		sig := fresh.OnBar(ctx, history[len(history)-1])
		if len(sig) != 1 {
			t.Fatalf("OnBar() signals = %d, want 1", len(sig))
		}
		if sig[0].TargetWeight != 0.0 {
			t.Errorf("TargetWeight = %v, want 0.0", sig[0].TargetWeight)
		}
	})

	// belowMeanHoldHistory builds a history whose final bar is still below
	// the mean (so the mean-touch exit would NOT fire) while long — the
	// setting in which only the time stop can trigger an exit.
	belowMeanHoldHistory := func() []domain.Bar {
		history := uptrendHistory(220, 100, 0.5)
		last := history[len(history)-1]
		stillLow := bsBar(len(history), last.Close-5)
		return append(history, stillLow)
	}

	t.Run("exit on time stop at exactly age == timeStop", func(t *testing.T) {
		bsInst := s.(*bollSnapback)
		fresh := &bollSnapback{smaPeriod: bsInst.smaPeriod, period: bsInst.period, k: bsInst.k, timeStop: bsInst.timeStop, entryWeight: bsInst.entryWeight}

		history := belowMeanHoldHistory()
		ctx := strategy.NewContext(history, longPortfolio(10, 90)).WithPositionAge(map[string]int{"T": fresh.timeStop})
		sig := fresh.OnBar(ctx, history[len(history)-1])
		if len(sig) != 1 {
			t.Fatalf("OnBar() signals = %d, want 1 (time stop)", len(sig))
		}
		if sig[0].TargetWeight != 0.0 {
			t.Errorf("TargetWeight = %v, want 0.0", sig[0].TargetWeight)
		}
	})

	t.Run("no time stop one bar before it elapses", func(t *testing.T) {
		bsInst := s.(*bollSnapback)
		fresh := &bollSnapback{smaPeriod: bsInst.smaPeriod, period: bsInst.period, k: bsInst.k, timeStop: bsInst.timeStop, entryWeight: bsInst.entryWeight}

		history := belowMeanHoldHistory()
		ctx := strategy.NewContext(history, longPortfolio(10, 90)).WithPositionAge(map[string]int{"T": fresh.timeStop - 1})
		sig := fresh.OnBar(ctx, history[len(history)-1])
		if len(sig) != 0 {
			t.Fatalf("OnBar() signals = %d, want 0 (age %d < timeStop %d)", len(sig), fresh.timeStop-1, fresh.timeStop)
		}
	})

	t.Run("unknown age (0) while holding never fires the time stop", func(t *testing.T) {
		bsInst := s.(*bollSnapback)
		fresh := &bollSnapback{smaPeriod: bsInst.smaPeriod, period: bsInst.period, k: bsInst.k, timeStop: bsInst.timeStop, entryWeight: bsInst.entryWeight}

		// Context built WITHOUT WithPositionAge: PositionAge returns 0 even
		// though the portfolio holds the position — the safe-degradation
		// convention says treat it as just-entered, never force an exit.
		history := belowMeanHoldHistory()
		ctx := strategy.NewContext(history, longPortfolio(10, 90))
		sig := fresh.OnBar(ctx, history[len(history)-1])
		if len(sig) != 0 {
			t.Fatalf("OnBar() signals = %d, want 0 (unknown age must not trigger the time stop)", len(sig))
		}
	})
}

func TestBollSnapback_WithParams(t *testing.T) {
	t.Run("valid full override applied", func(t *testing.T) {
		base := newBollSnapback()
		got, err := base.WithParams(map[string]float64{"period": 10, "k": 1.5, "timeStop": 5})
		if err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		bs, ok := got.(*bollSnapback)
		if !ok {
			t.Fatalf("WithParams() returned %T, want *bollSnapback", got)
		}
		if bs.period != 10 || bs.k != 1.5 || bs.timeStop != 5 {
			t.Errorf("period/k/timeStop = %d/%v/%d, want 10/1.5/5", bs.period, bs.k, bs.timeStop)
		}
	})

	t.Run("partial map keeps defaults", func(t *testing.T) {
		base := newBollSnapback()
		got, err := base.WithParams(map[string]float64{"k": 2.5})
		if err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		bs := got.(*bollSnapback)
		if bs.k != 2.5 {
			t.Errorf("k = %v, want 2.5", bs.k)
		}
		if bs.period != 20 {
			t.Errorf("period = %d, want default 20", bs.period)
		}
		if bs.timeStop != 10 {
			t.Errorf("timeStop = %d, want default 10", bs.timeStop)
		}
	})

	t.Run("unknown key rejected", func(t *testing.T) {
		base := newBollSnapback()
		if _, err := base.WithParams(map[string]float64{"bogus": 1}); err == nil {
			t.Fatal("WithParams() error = nil, want error for unknown key")
		}
	})

	t.Run("period < 2 rejected", func(t *testing.T) {
		base := newBollSnapback()
		if _, err := base.WithParams(map[string]float64{"period": 1}); err == nil {
			t.Fatal("WithParams() error = nil, want error for period<2")
		}
	})

	t.Run("non-integral period rejected", func(t *testing.T) {
		base := newBollSnapback()
		if _, err := base.WithParams(map[string]float64{"period": 10.5}); err == nil {
			t.Fatal("WithParams() error = nil, want error for non-integral period")
		}
	})

	t.Run("k <= 0 rejected", func(t *testing.T) {
		base := newBollSnapback()
		if _, err := base.WithParams(map[string]float64{"k": 0}); err == nil {
			t.Fatal("WithParams() error = nil, want error for k<=0")
		}
	})

	t.Run("timeStop < 1 rejected", func(t *testing.T) {
		base := newBollSnapback()
		if _, err := base.WithParams(map[string]float64{"timeStop": 0}); err == nil {
			t.Fatal("WithParams() error = nil, want error for timeStop<1")
		}
	})

	t.Run("non-integral timeStop rejected", func(t *testing.T) {
		base := newBollSnapback()
		if _, err := base.WithParams(map[string]float64{"timeStop": 5.5}); err == nil {
			t.Fatal("WithParams() error = nil, want error for non-integral timeStop")
		}
	})

	t.Run("receiver not mutated", func(t *testing.T) {
		base := newBollSnapback()
		if _, err := base.WithParams(map[string]float64{"period": 10, "k": 1.5, "timeStop": 5}); err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		if base.period != 20 || base.k != 2.0 || base.timeStop != 10 {
			t.Errorf("receiver mutated: period/k/timeStop = %d/%v/%d, want unchanged 20/2.0/10", base.period, base.k, base.timeStop)
		}
	})

	t.Run("ParamSpace has expected shape", func(t *testing.T) {
		defs := newBollSnapback().ParamSpace()
		if len(defs) != 3 {
			t.Fatalf("len(ParamSpace()) = %d, want 3", len(defs))
		}
	})
}

var _ strategy.Tunable = (*bollSnapback)(nil)
