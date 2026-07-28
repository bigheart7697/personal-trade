package strategies

import (
	"testing"
	"time"

	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// ibsBar builds an ibs test bar for symbol "I" on day `day` with explicit
// OHLC values (unlike most other strategies in this package, ibs genuinely
// needs high/low, not just close).
func ibsBar(day int, o, h, l, c float64) domain.Bar {
	return domain.Bar{
		Symbol: "I",
		Time:   time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC).AddDate(0, 0, day),
		Open:   o, High: h, Low: l, Close: c,
		Volume: 1000,
	}
}

// ibsUptrendHistory builds n bars of a steadily rising close series whose
// IBS is always exactly 0.5 (close dead-center in the bar's own range) —
// a clean uptrend regime (close > SMA200 once warmed up) with no IBS
// entry/exit noise of its own, meant as a base onto which specific
// IBS-triggering bars are appended.
func ibsUptrendHistory(n int, start, step float64) []domain.Bar {
	bars := make([]domain.Bar, n)
	for i := 0; i < n; i++ {
		c := start + float64(i)*step
		bars[i] = ibsBar(i, c, c+5, c-5, c) // IBS = 0.5
	}
	return bars
}

// ibsDowntrendHistory is ibsUptrendHistory's falling mirror, for the
// SMA200-filter test (trend filter should block entries below SMA200
// regardless of IBS).
func ibsDowntrendHistory(n int, start, step float64) []domain.Bar {
	bars := make([]domain.Bar, n)
	for i := 0; i < n; i++ {
		c := start - float64(i)*step
		bars[i] = ibsBar(i, c, c+5, c-5, c) // IBS = 0.5
	}
	return bars
}

// ibsEntryDip returns a bar to append immediately after history's last bar,
// engineered to have a low IBS (weak close near the bar's own low) RELATIVE
// TO WHEREVER THE SERIES CURRENTLY SITS — not hardcoded to an absolute
// price level, which would fail the SMA200 trend filter once a long
// uptrend has climbed well above its starting price.
func ibsEntryDip(history []domain.Bar) domain.Bar {
	last := history[len(history)-1].Close
	return ibsBar(len(history), last, last+1, last-11, last-10) // ibs = 1/12 ~= 0.083
}

// ibsStrongCloseBar returns a bar (relative to the series' current level)
// whose close sits at the top of its own range: IBS = 1.0, decisively above
// any exitAbove threshold in the grid.
func ibsStrongCloseBar(history []domain.Bar) domain.Bar {
	last := history[len(history)-1].Close
	return ibsBar(len(history), last, last+5, last-1, last+5) // close == high -> IBS = 1.0
}

// ibsNeutralBar returns a bar (relative to the series' current level) whose
// IBS is exactly 0.5 — strictly between any enterBelow/exitAbove band in
// the grid, i.e. never decisive.
func ibsNeutralBar(history []domain.Bar) domain.Bar {
	last := history[len(history)-1].Close
	return ibsBar(len(history), last, last+10, last-10, last) // IBS = 0.5
}

func TestIBS_Determinism(t *testing.T) {
	history := ibsUptrendHistory(220, 100, 0.5)
	bars := append(history, ibsEntryDip(history))

	s1 := &ibs{enterBelow: 0.15, exitAbove: 0.85}
	s2 := &ibs{enterBelow: 0.15, exitAbove: 0.85}

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

func TestIBS_NoLookahead(t *testing.T) {
	history := ibsUptrendHistory(220, 100, 0.5)
	base := append(history, ibsEntryDip(history))

	// Extend with 50 more (calm, neutral-IBS) uptrend bars so the longer run
	// has a strictly larger bar set to compare prefixes against.
	extra := ibsUptrendHistory(50, base[len(base)-1].Close+1, 0.5)
	for i := range extra {
		extra[i].Time = base[len(base)-1].Time.AddDate(0, 0, i+1)
	}
	longBars := append(append([]domain.Bar{}, base...), extra...)

	sShort := &ibs{enterBelow: 0.15, exitAbove: 0.85}
	sLong := &ibs{enterBelow: 0.15, exitAbove: 0.85}

	sigsShort := runMLStrategy(sShort, base)
	sigsLong := runMLStrategy(sLong, longBars)

	if len(sigsLong) < len(sigsShort) {
		t.Fatalf("longer run produced fewer signals (%d) than the shorter run (%d)", len(sigsLong), len(sigsShort))
	}
	for i := range sigsShort {
		if !signalsEqual(sigsShort[i], sigsLong[i]) {
			t.Fatalf("signal at index %d changed when 50 future bars were appended: %+v (short) vs %+v (long) — lookahead leak", i, sigsShort[i], sigsLong[i])
		}
	}
}

// TestIBS_TargetWeights_EntersOnWeakCloseInUptrend builds an uptrend (close
// > SMA200) with a sharp one-bar dip whose close sits near the bar's own
// low (IBS well below enterBelow) and checks the strategy wants full
// weight.
func TestIBS_TargetWeights_EntersOnWeakCloseInUptrend(t *testing.T) {
	s := &ibs{enterBelow: 0.15, exitAbove: 0.85}

	history := ibsUptrendHistory(220, 100, 0.5)
	dip := ibsEntryDip(history)
	history = append(history, dip)

	if v := ibsValue(dip); v > s.enterBelow {
		t.Fatalf("test setup invalid: ibsValue(dip) = %v, want <= enterBelow %v", v, s.enterBelow)
	}

	ctx := strategy.NewContext(history, flatPortfolio())
	weights := s.TargetWeights(ctx)
	if got, want := weights["I"], 1.0; got != want {
		t.Errorf("TargetWeights()[I] = %v, want %v (weak close in an uptrend)", got, want)
	}
}

// TestIBS_TargetWeights_NoEntryBelowSMA200 repeats the same weak-close setup
// but in a downtrend (close < SMA200): the trend filter must block entry
// regardless of how oversold IBS is.
func TestIBS_TargetWeights_NoEntryBelowSMA200(t *testing.T) {
	s := &ibs{enterBelow: 0.15, exitAbove: 0.85}

	history := ibsDowntrendHistory(220, 300, 0.5)
	dip := ibsEntryDip(history)
	history = append(history, dip)

	if v := ibsValue(dip); v > s.enterBelow {
		t.Fatalf("test setup invalid: ibsValue(dip) = %v, want <= enterBelow %v", v, s.enterBelow)
	}

	ctx := strategy.NewContext(history, flatPortfolio())
	weights := s.TargetWeights(ctx)
	if len(weights) != 0 {
		t.Errorf("TargetWeights() = %+v, want empty map (regime filter should block entry in a downtrend)", weights)
	}
}

// TestIBS_TargetWeights_ExitsOnStrongClose builds a history where the
// strategy is (per the bounded backward replay) currently long, then adds a
// strong-close bar (IBS >= exitAbove) and checks the strategy wants nothing.
func TestIBS_TargetWeights_ExitsOnStrongClose(t *testing.T) {
	s := &ibs{enterBelow: 0.15, exitAbove: 0.85}

	history := ibsUptrendHistory(220, 100, 0.5)
	history = append(history, ibsEntryDip(history))

	// Confirm the entry actually fired on this crafted history (test setup
	// sanity), then append a strong-close bar.
	if w := s.TargetWeights(strategy.NewContext(history, flatPortfolio()))["I"]; w != 1.0 {
		t.Fatalf("test setup invalid: entry weight = %v, want 1.0", w)
	}
	history = append(history, ibsStrongCloseBar(history))

	ctx := strategy.NewContext(history, flatPortfolio())
	weights := s.TargetWeights(ctx)
	if len(weights) != 0 {
		t.Errorf("TargetWeights() = %+v, want empty map after a strong-close exit", weights)
	}
}

// TestIBS_TargetWeights_HoldsBetweenBands enters on a weak close, then adds
// a handful of bars whose IBS sits strictly between enterBelow and
// exitAbove (neither decisive) — the bounded backward replay should keep
// the position held.
func TestIBS_TargetWeights_HoldsBetweenBands(t *testing.T) {
	s := &ibs{enterBelow: 0.15, exitAbove: 0.85}

	history := ibsUptrendHistory(220, 100, 0.5)
	history = append(history, ibsEntryDip(history))

	for i := 0; i < 3; i++ {
		history = append(history, ibsNeutralBar(history))
	}

	ctx := strategy.NewContext(history, flatPortfolio())
	weights := s.TargetWeights(ctx)
	if got, want := weights["I"], 1.0; got != want {
		t.Errorf("TargetWeights()[I] = %v, want %v (held between the bands, well under the time stop)", got, want)
	}
}

// TestIBS_TargetWeights_TimeStopFires is HoldsBetweenBands but extended past
// ibsMaxHold(10) neutral bars — the hard time-stop must flatten the
// position even though neither IBS band ever fired again.
func TestIBS_TargetWeights_TimeStopFires(t *testing.T) {
	s := &ibs{enterBelow: 0.15, exitAbove: 0.85}

	history := ibsUptrendHistory(220, 100, 0.5)
	history = append(history, ibsEntryDip(history))

	for i := 0; i < ibsMaxHold+2; i++ {
		history = append(history, ibsNeutralBar(history))
	}

	ctx := strategy.NewContext(history, flatPortfolio())
	weights := s.TargetWeights(ctx)
	if len(weights) != 0 {
		t.Errorf("TargetWeights() = %+v, want empty map once held age reaches the hard time stop", weights)
	}
}

func TestIBS_IBSValue(t *testing.T) {
	t.Run("guards a zero-range bar to neutral 0.5", func(t *testing.T) {
		bar := ibsBar(0, 100, 100, 100, 100)
		if got, want := ibsValue(bar), 0.5; got != want {
			t.Errorf("ibsValue() = %v, want %v for a zero-range bar", got, want)
		}
	})

	t.Run("computes (close-low)/(high-low)", func(t *testing.T) {
		bar := ibsBar(0, 100, 110, 90, 95) // (95-90)/(110-90) = 0.25
		if got, want := ibsValue(bar), 0.25; got != want {
			t.Errorf("ibsValue() = %v, want %v", got, want)
		}
	})
}

func TestIBS_TargetWeights_PurityFromPortfolio(t *testing.T) {
	history := ibsUptrendHistory(220, 100, 0.5)
	history = append(history, ibsEntryDip(history))

	s1 := &ibs{enterBelow: 0.15, exitAbove: 0.85}
	s2 := &ibs{enterBelow: 0.15, exitAbove: 0.85}

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

func TestIBS_WarmupBars(t *testing.T) {
	s := newIBS()
	if got, want := s.WarmupBars(), 201; got != want {
		t.Errorf("WarmupBars() = %d, want %d", got, want)
	}
}

func TestIBS_WithParams(t *testing.T) {
	base := newIBS()

	t.Run("valid full override applied", func(t *testing.T) {
		got, err := base.WithParams(map[string]float64{"enterBelow": 0.25, "exitAbove": 0.75})
		if err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		i, ok := got.(*ibs)
		if !ok {
			t.Fatalf("WithParams() returned %T, want *ibs", got)
		}
		if i.enterBelow != 0.25 || i.exitAbove != 0.75 {
			t.Errorf("enterBelow/exitAbove = %v/%v, want 0.25/0.75", i.enterBelow, i.exitAbove)
		}
	})

	t.Run("partial map keeps defaults", func(t *testing.T) {
		got, err := base.WithParams(map[string]float64{"enterBelow": 0.25})
		if err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		i := got.(*ibs)
		if i.enterBelow != 0.25 {
			t.Errorf("enterBelow = %v, want 0.25", i.enterBelow)
		}
		if i.exitAbove != 0.85 {
			t.Errorf("exitAbove = %v, want default 0.85", i.exitAbove)
		}
	})

	t.Run("unknown key rejected", func(t *testing.T) {
		if _, err := base.WithParams(map[string]float64{"bogus": 1}); err == nil {
			t.Fatal("WithParams() error = nil, want error for unknown key")
		}
	})

	t.Run("enterBelow out of range rejected", func(t *testing.T) {
		if _, err := base.WithParams(map[string]float64{"enterBelow": 0}); err == nil {
			t.Fatal("WithParams() error = nil, want error for enterBelow<=0")
		}
		if _, err := base.WithParams(map[string]float64{"enterBelow": 1}); err == nil {
			t.Fatal("WithParams() error = nil, want error for enterBelow>=1")
		}
	})

	t.Run("exitAbove out of range rejected", func(t *testing.T) {
		if _, err := base.WithParams(map[string]float64{"exitAbove": 0}); err == nil {
			t.Fatal("WithParams() error = nil, want error for exitAbove<=0")
		}
		if _, err := base.WithParams(map[string]float64{"exitAbove": 1}); err == nil {
			t.Fatal("WithParams() error = nil, want error for exitAbove>=1")
		}
	})

	t.Run("enterBelow >= exitAbove rejected", func(t *testing.T) {
		if _, err := base.WithParams(map[string]float64{"enterBelow": 0.9, "exitAbove": 0.8}); err == nil {
			t.Fatal("WithParams() error = nil, want error for enterBelow>=exitAbove")
		}
	})

	t.Run("receiver not mutated", func(t *testing.T) {
		if _, err := base.WithParams(map[string]float64{"enterBelow": 0.25, "exitAbove": 0.75}); err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		if base.enterBelow != 0.15 || base.exitAbove != 0.85 {
			t.Errorf("receiver mutated: enterBelow/exitAbove = %v/%v, want unchanged 0.15/0.85", base.enterBelow, base.exitAbove)
		}
	})

	t.Run("ParamSpace/Grid has the documented 4 combos", func(t *testing.T) {
		defs := newIBS().ParamSpace()
		combos := strategy.Grid(defs)
		if len(combos) != 4 {
			t.Fatalf("len(Grid(ParamSpace())) = %d, want 4", len(combos))
		}
	})
}

var _ strategy.Tunable = (*ibs)(nil)
var _ strategy.TargetWeighter = (*ibs)(nil)
