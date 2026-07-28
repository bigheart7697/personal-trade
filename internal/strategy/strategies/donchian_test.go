package strategies

import (
	"testing"
	"time"

	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// dBar builds a Donchian test bar for symbol "T" on day `day` with the given
// OHLC values.
func dBar(day int, o, h, l, c float64) domain.Bar {
	return domain.Bar{
		Symbol: "T",
		Time:   time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC).AddDate(0, 0, day),
		Open:   o, High: h, Low: l, Close: c,
		Volume: 1000,
	}
}

func flatPortfolio() *domain.Portfolio {
	return domain.NewPortfolio(100000)
}

func longPortfolio(qty int64, avgPrice float64) *domain.Portfolio {
	p := domain.NewPortfolio(100000)
	p.Positions["T"] = domain.Position{Symbol: "T", Qty: qty, AvgPrice: avgPrice}
	return p
}

func TestDonchian_OnBar(t *testing.T) {
	s, err := newDonchian().WithParams(map[string]float64{"entry": 5, "exit": 3})
	if err != nil {
		t.Fatalf("WithParams() error = %v", err)
	}

	t.Run("entry breakout while flat", func(t *testing.T) {
		// Prior 5 bars have highs 10,11,12,13,14 -> channel high = 14.
		// Today's close (15) breaks above it.
		history := []domain.Bar{
			dBar(0, 9, 10, 8, 10),
			dBar(1, 10, 11, 9, 11),
			dBar(2, 11, 12, 10, 12),
			dBar(3, 12, 13, 11, 13),
			dBar(4, 13, 14, 12, 14),
			dBar(5, 14, 16, 14, 15),
		}
		ctx := strategy.NewContext(history, flatPortfolio())
		sig := s.OnBar(ctx, history[len(history)-1])
		if len(sig) != 1 {
			t.Fatalf("OnBar() signals = %d, want 1", len(sig))
		}
		if sig[0].TargetWeight != 1.0 {
			t.Errorf("TargetWeight = %v, want 1.0", sig[0].TargetWeight)
		}
	})

	t.Run("no entry when close equals prior high", func(t *testing.T) {
		// Channel high = 14 (from bars 0..4). Today's close == 14 exactly:
		// must NOT trigger (breakout requires strictly greater).
		history := []domain.Bar{
			dBar(0, 9, 10, 8, 10),
			dBar(1, 10, 11, 9, 11),
			dBar(2, 11, 12, 10, 12),
			dBar(3, 12, 13, 11, 13),
			dBar(4, 13, 14, 12, 14),
			dBar(5, 13, 14, 13, 14),
		}
		ctx := strategy.NewContext(history, flatPortfolio())
		sig := s.OnBar(ctx, history[len(history)-1])
		if len(sig) != 0 {
			t.Fatalf("OnBar() signals = %d, want 0 (close==prior high must not enter)", len(sig))
		}
	})

	t.Run("exit on breakdown while long", func(t *testing.T) {
		// Prior 3 bars have lows 20,19,18 -> channel low = 18.
		// Today's close (17) breaks below it.
		history := []domain.Bar{
			dBar(0, 22, 23, 20, 21),
			dBar(1, 21, 22, 19, 20),
			dBar(2, 20, 21, 18, 19),
			dBar(3, 19, 20, 17, 17),
		}
		ctx := strategy.NewContext(history, longPortfolio(10, 20))
		sig := s.OnBar(ctx, history[len(history)-1])
		if len(sig) != 1 {
			t.Fatalf("OnBar() signals = %d, want 1", len(sig))
		}
		if sig[0].TargetWeight != 0.0 {
			t.Errorf("TargetWeight = %v, want 0.0", sig[0].TargetWeight)
		}
	})

	t.Run("no signal mid-channel", func(t *testing.T) {
		// Flat: channel high (bars 0..4) = 14; today's close 13 doesn't
		// break out.
		history := []domain.Bar{
			dBar(0, 9, 10, 8, 10),
			dBar(1, 10, 11, 9, 11),
			dBar(2, 11, 12, 10, 12),
			dBar(3, 12, 13, 11, 13),
			dBar(4, 13, 14, 12, 14),
			dBar(5, 13, 13, 12, 13),
		}
		ctx := strategy.NewContext(history, flatPortfolio())
		sig := s.OnBar(ctx, history[len(history)-1])
		if len(sig) != 0 {
			t.Fatalf("OnBar() signals = %d, want 0 (mid-channel)", len(sig))
		}

		// Long: channel low (bars 0..2) = 18; today's close 19 doesn't
		// break down.
		historyLong := []domain.Bar{
			dBar(0, 22, 23, 20, 21),
			dBar(1, 21, 22, 19, 20),
			dBar(2, 20, 21, 18, 19),
			dBar(3, 19, 21, 19, 19.5),
		}
		ctxLong := strategy.NewContext(historyLong, longPortfolio(10, 20))
		sigLong := s.OnBar(ctxLong, historyLong[len(historyLong)-1])
		if len(sigLong) != 0 {
			t.Fatalf("OnBar() signals = %d, want 0 (mid-channel, long)", len(sigLong))
		}
	})

	t.Run("current bar's own high cannot create the breakout channel", func(t *testing.T) {
		// Prior channel high (bars 0..4) = 14. Today's bar has a huge spike
		// high of 100, but close (13.5) stays below the PRIOR channel high
		// (14). The exclude-current-bar rule means today's spike high must
		// not be counted as part of the channel used to judge today's own
		// breakout, so no entry should fire.
		history := []domain.Bar{
			dBar(0, 9, 10, 8, 10),
			dBar(1, 10, 11, 9, 11),
			dBar(2, 11, 12, 10, 12),
			dBar(3, 12, 13, 11, 13),
			dBar(4, 13, 14, 12, 14),
			dBar(5, 13, 100, 13, 13.5),
		}
		ctx := strategy.NewContext(history, flatPortfolio())
		sig := s.OnBar(ctx, history[len(history)-1])
		if len(sig) != 0 {
			t.Fatalf("OnBar() signals = %d, want 0 (current bar's high must not build its own channel)", len(sig))
		}
	})
}

func TestDonchian_WarmupBars(t *testing.T) {
	tests := []struct {
		name        string
		entry, exit int
		want        int
	}{
		{name: "entry >= exit", entry: 55, exit: 20, want: 56},
		{name: "entry == exit", entry: 10, exit: 10, want: 11},
		{name: "exit > entry", entry: 10, exit: 30, want: 31},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s := &donchian{entry: tc.entry, exit: tc.exit}
			if got := s.WarmupBars(); got != tc.want {
				t.Errorf("WarmupBars() = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestDonchian_WithParams(t *testing.T) {
	t.Run("valid full override applied", func(t *testing.T) {
		base := newDonchian()
		got, err := base.WithParams(map[string]float64{"entry": 40, "exit": 10})
		if err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		d, ok := got.(*donchian)
		if !ok {
			t.Fatalf("WithParams() returned %T, want *donchian", got)
		}
		if d.entry != 40 || d.exit != 10 {
			t.Errorf("entry/exit = %d/%d, want 40/10", d.entry, d.exit)
		}
	})

	t.Run("partial map keeps defaults", func(t *testing.T) {
		base := newDonchian()
		got, err := base.WithParams(map[string]float64{"entry": 70})
		if err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		d := got.(*donchian)
		if d.entry != 70 {
			t.Errorf("entry = %d, want 70", d.entry)
		}
		if d.exit != 20 {
			t.Errorf("exit = %d, want default 20", d.exit)
		}
	})

	t.Run("unknown key rejected", func(t *testing.T) {
		base := newDonchian()
		if _, err := base.WithParams(map[string]float64{"bogus": 1}); err == nil {
			t.Fatal("WithParams() error = nil, want error for unknown key")
		}
	})

	t.Run("entry <= exit rejected", func(t *testing.T) {
		base := newDonchian()
		if _, err := base.WithParams(map[string]float64{"entry": 10, "exit": 20}); err == nil {
			t.Fatal("WithParams() error = nil, want error for entry<=exit")
		}
	})

	t.Run("entry equal to exit rejected", func(t *testing.T) {
		base := newDonchian()
		if _, err := base.WithParams(map[string]float64{"entry": 20, "exit": 20}); err == nil {
			t.Fatal("WithParams() error = nil, want error for entry==exit")
		}
	})

	t.Run("entry < 2 rejected", func(t *testing.T) {
		base := newDonchian()
		if _, err := base.WithParams(map[string]float64{"entry": 1, "exit": -5}); err == nil {
			t.Fatal("WithParams() error = nil, want error for entry<2")
		}
	})

	t.Run("exit < 2 rejected", func(t *testing.T) {
		base := newDonchian()
		if _, err := base.WithParams(map[string]float64{"exit": 1}); err == nil {
			t.Fatal("WithParams() error = nil, want error for exit<2")
		}
	})

	t.Run("non-integral value rejected", func(t *testing.T) {
		base := newDonchian()
		if _, err := base.WithParams(map[string]float64{"entry": 55.5}); err == nil {
			t.Fatal("WithParams() error = nil, want error for non-integral entry")
		}
	})

	t.Run("receiver not mutated", func(t *testing.T) {
		base := newDonchian()
		if _, err := base.WithParams(map[string]float64{"entry": 40, "exit": 10}); err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		if base.entry != 55 || base.exit != 20 {
			t.Errorf("receiver mutated: entry/exit = %d/%d, want unchanged 55/20", base.entry, base.exit)
		}
	})

	t.Run("ParamSpace has expected shape", func(t *testing.T) {
		defs := newDonchian().ParamSpace()
		if len(defs) != 2 {
			t.Fatalf("len(ParamSpace()) = %d, want 2", len(defs))
		}
	})
}

func TestDonchian_TargetWeights_AgreesWithBreakoutEntry(t *testing.T) {
	s, err := newDonchian().WithParams(map[string]float64{"entry": 5, "exit": 3})
	if err != nil {
		t.Fatalf("WithParams() error = %v", err)
	}
	d := s.(*donchian)

	// Same breakout history as TestDonchian_OnBar's "entry breakout while
	// flat" subtest: prior 5 bars have highs 10..14, today's close (15)
	// breaks out.
	history := []domain.Bar{
		dBar(0, 9, 10, 8, 10),
		dBar(1, 10, 11, 9, 11),
		dBar(2, 11, 12, 10, 12),
		dBar(3, 12, 13, 11, 13),
		dBar(4, 13, 14, 12, 14),
		dBar(5, 14, 16, 14, 15),
	}
	ctx := strategy.NewContext(history, flatPortfolio())

	// Confirm OnBar agrees this is an entry on the same history.
	onBarSigs := d.OnBar(ctx, history[len(history)-1])
	if len(onBarSigs) != 1 || onBarSigs[0].TargetWeight != 1.0 {
		t.Fatalf("test setup invalid: OnBar() = %+v, want a breakout entry signal", onBarSigs)
	}

	weights := d.TargetWeights(ctx)
	if got, want := weights["T"], 1.0; got != want {
		t.Errorf("TargetWeights()[T] = %v, want %v (agrees with OnBar's breakout entry)", got, want)
	}
}

func TestDonchian_TargetWeights_ReplaysExitAfterBreakdown(t *testing.T) {
	s, err := newDonchian().WithParams(map[string]float64{"entry": 5, "exit": 3})
	if err != nil {
		t.Fatalf("WithParams() error = %v", err)
	}
	d := s.(*donchian)

	// Extend the breakout history with a subsequent breakdown: the 3-bar
	// exit channel prior to bar 6 (bars 3,4,5: lows 11,12,14) has a low of
	// 11; bar 6's close (9) breaks below it, so the replayed state machine
	// must end flat.
	history := []domain.Bar{
		dBar(0, 9, 10, 8, 10),
		dBar(1, 10, 11, 9, 11),
		dBar(2, 11, 12, 10, 12),
		dBar(3, 12, 13, 11, 13),
		dBar(4, 13, 14, 12, 14),
		dBar(5, 14, 16, 14, 15), // breaks out, enters
		dBar(6, 15, 15, 10, 9),  // breaks down, exits
	}
	ctx := strategy.NewContext(history, flatPortfolio())

	weights := d.TargetWeights(ctx)
	if len(weights) != 0 {
		t.Errorf("TargetWeights() = %+v, want empty map after the replayed breakout is followed by a breakdown", weights)
	}
}

func TestDonchian_TargetWeights_InsufficientHistory(t *testing.T) {
	s := newDonchian()
	history := []domain.Bar{
		dBar(0, 9, 10, 8, 10),
		dBar(1, 10, 11, 9, 11),
	}
	ctx := strategy.NewContext(history, flatPortfolio())

	weights := s.TargetWeights(ctx)
	if len(weights) != 0 {
		t.Errorf("TargetWeights() = %+v, want empty map when history is shorter than WarmupBars", weights)
	}
}

var _ strategy.Tunable = (*donchian)(nil)
var _ strategy.TargetWeighter = (*donchian)(nil)
