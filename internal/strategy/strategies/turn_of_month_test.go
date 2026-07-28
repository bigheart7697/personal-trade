package strategies

import (
	"testing"
	"time"

	"tradeforge/internal/domain"
	"tradeforge/internal/strategy"
)

// tomBar builds a turn-of-month test bar for symbol "T" on the given
// calendar date (OHLC don't matter — this strategy is pure calendar logic).
func tomBar(date time.Time) domain.Bar {
	return domain.Bar{
		Symbol: "T",
		Time:   date,
		Open:   100, High: 100, Low: 100, Close: 100,
		Volume: 1000,
	}
}

// tomBusinessDays builds n consecutive weekday (Mon-Fri) bars starting at
// start, skipping weekends — so "trading day index within month" behaves
// like a real calendar rather than every single day counting.
func tomBusinessDays(start time.Time, n int) []domain.Bar {
	bars := make([]domain.Bar, 0, n)
	d := start
	for len(bars) < n {
		if d.Weekday() != time.Saturday && d.Weekday() != time.Sunday {
			bars = append(bars, tomBar(d))
		}
		d = d.AddDate(0, 0, 1)
	}
	return bars
}

// tomBusinessDaysThrough builds weekday-only bars for every date from start
// through end (inclusive), skipping weekends.
func tomBusinessDaysThrough(start, end time.Time) []domain.Bar {
	var bars []domain.Bar
	for d := start; !d.After(end); d = d.AddDate(0, 0, 1) {
		if d.Weekday() != time.Saturday && d.Weekday() != time.Sunday {
			bars = append(bars, tomBar(d))
		}
	}
	return bars
}

func TestTurnOfMonth_Determinism(t *testing.T) {
	bars := tomBusinessDays(time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC), 120)

	s1 := &turnOfMonth{entryDay: 26, exitAfter: 3}
	s2 := &turnOfMonth{entryDay: 26, exitAfter: 3}

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

func TestTurnOfMonth_NoLookahead(t *testing.T) {
	bars := tomBusinessDays(time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC), 160)

	n := 100
	sShort := &turnOfMonth{entryDay: 26, exitAfter: 3}
	sLong := &turnOfMonth{entryDay: 26, exitAfter: 3}

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

// TestTurnOfMonth_TargetWeights_LongOnMonthEnd builds history ending on
// January 26, 2024 (on/after the default entryDay=26), and checks the
// strategy wants full weight for month-end positioning.
func TestTurnOfMonth_TargetWeights_LongOnMonthEnd(t *testing.T) {
	s := newTurnOfMonth()

	// Jan 26, 2024 is a Friday.
	history := tomBusinessDaysThrough(
		time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC),
		time.Date(2024, 1, 26, 0, 0, 0, 0, time.UTC),
	)
	last := history[len(history)-1].Time
	if last.Day() != 26 {
		t.Fatalf("test setup invalid: last bar day = %d, want 26", last.Day())
	}

	ctx := strategy.NewContext(history, flatPortfolio())
	weights := s.TargetWeights(ctx)
	if got, want := weights["T"], 1.0; got != want {
		t.Errorf("TargetWeights()[T] = %v, want %v (on/after entryDay)", got, want)
	}
}

// TestTurnOfMonth_TargetWeights_LongOnSecondTradingDayOfMonth builds a
// history whose last bar is the 2nd trading day of its month (within the
// default exitAfter=3 window) and checks the strategy wants full weight.
func TestTurnOfMonth_TargetWeights_LongOnSecondTradingDayOfMonth(t *testing.T) {
	s := newTurnOfMonth()

	// Feb 1, 2024 is a Thursday; Feb 2 is the 2nd trading day of February.
	history := tomBusinessDays(time.Date(2024, 2, 1, 0, 0, 0, 0, time.UTC), 2)
	idx := tradingDayIndexInMonth(history)
	if idx != 2 {
		t.Fatalf("test setup invalid: tradingDayIndexInMonth = %d, want 2", idx)
	}

	ctx := strategy.NewContext(history, flatPortfolio())
	weights := s.TargetWeights(ctx)
	if got, want := weights["T"], 1.0; got != want {
		t.Errorf("TargetWeights()[T] = %v, want %v (2nd trading day, within exitAfter)", got, want)
	}
}

// TestTurnOfMonth_TargetWeights_FlatMidMonth builds a history whose last bar
// sits comfortably mid-month (well past exitAfter, well before entryDay)
// and checks the strategy wants nothing.
func TestTurnOfMonth_TargetWeights_FlatMidMonth(t *testing.T) {
	s := newTurnOfMonth()

	// Jan 15, 2024: the 15th, well before entryDay=26, and (with January
	// starting on a Monday) well past the first exitAfter=3 trading days.
	history := tomBusinessDays(time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC), 11)
	last := history[len(history)-1].Time
	if last.Day() >= s.entryDay {
		t.Fatalf("test setup invalid: last bar day = %d, want < entryDay %d", last.Day(), s.entryDay)
	}
	if idx := tradingDayIndexInMonth(history); idx <= s.exitAfter {
		t.Fatalf("test setup invalid: tradingDayIndexInMonth = %d, want > exitAfter %d", idx, s.exitAfter)
	}

	ctx := strategy.NewContext(history, flatPortfolio())
	weights := s.TargetWeights(ctx)
	if len(weights) != 0 {
		t.Errorf("TargetWeights() = %+v, want empty map mid-month", weights)
	}
}

// TestTurnOfMonth_TradingDayIndexInMonth_ResetsAcrossMonthBoundary checks
// the pure-history trading-day counter itself: it must count only bars
// sharing the CURRENT bar's year+month, resetting to 1 the first trading
// day of a new month regardless of how many bars preceded it.
func TestTurnOfMonth_TradingDayIndexInMonth_ResetsAcrossMonthBoundary(t *testing.T) {
	// The last week and a half of January 2024, then Feb 1 (a Thursday) —
	// the first trading day of February. Regardless of how many trailing
	// January bars precede it, the index must reset to 1.
	history := tomBusinessDaysThrough(
		time.Date(2024, 1, 20, 0, 0, 0, 0, time.UTC),
		time.Date(2024, 1, 31, 0, 0, 0, 0, time.UTC),
	)
	history = append(history, tomBar(time.Date(2024, 2, 1, 0, 0, 0, 0, time.UTC)))

	if idx := tradingDayIndexInMonth(history); idx != 1 {
		t.Errorf("tradingDayIndexInMonth() = %d, want 1 on the first trading day of a new month", idx)
	}
}

func TestTurnOfMonth_OnBar_EntersAndExitsOnTheEdge(t *testing.T) {
	s := newTurnOfMonth()

	t.Run("enters while flat when target is 1.0", func(t *testing.T) {
		// Jan 26, 2024 is a Friday, on/after the default entryDay=26.
		history := tomBusinessDaysThrough(
			time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC),
			time.Date(2024, 1, 26, 0, 0, 0, 0, time.UTC),
		)
		ctx := strategy.NewContext(history, flatPortfolio())
		sig := s.OnBar(ctx, history[len(history)-1])
		if len(sig) != 1 || sig[0].TargetWeight != 1.0 {
			t.Fatalf("OnBar() = %+v, want a single entry signal at weight 1.0", sig)
		}
	})

	t.Run("no re-entry signal while already long and still in the window", func(t *testing.T) {
		// Jan 29, 2024 is a Monday, still on/after entryDay=26.
		history := tomBusinessDaysThrough(
			time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC),
			time.Date(2024, 1, 29, 0, 0, 0, 0, time.UTC),
		)
		ctx := strategy.NewContext(history, longPortfolio(10, 100))
		sig := s.OnBar(ctx, history[len(history)-1])
		if len(sig) != 0 {
			t.Fatalf("OnBar() = %+v, want no signal (already long, still in the window)", sig)
		}
	})

	t.Run("exits while long when target flattens mid-month", func(t *testing.T) {
		history := tomBusinessDays(time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC), 11) // mid-month, see FlatMidMonth test
		ctx := strategy.NewContext(history, longPortfolio(10, 100))
		sig := s.OnBar(ctx, history[len(history)-1])
		if len(sig) != 1 || sig[0].TargetWeight != 0.0 {
			t.Fatalf("OnBar() = %+v, want a single exit signal at weight 0.0", sig)
		}
	})
}

func TestTurnOfMonth_TargetWeights_PurityFromPortfolio(t *testing.T) {
	history := tomBusinessDays(time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC), 30)

	s1 := newTurnOfMonth()
	s2 := newTurnOfMonth()

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

func TestTurnOfMonth_WarmupBars(t *testing.T) {
	s := newTurnOfMonth()
	if got, want := s.WarmupBars(), 25; got != want {
		t.Errorf("WarmupBars() = %d, want %d", got, want)
	}
}

func TestTurnOfMonth_WithParams(t *testing.T) {
	base := newTurnOfMonth()

	t.Run("valid full override applied", func(t *testing.T) {
		got, err := base.WithParams(map[string]float64{"entryDay": 24, "exitAfter": 5})
		if err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		tom, ok := got.(*turnOfMonth)
		if !ok {
			t.Fatalf("WithParams() returned %T, want *turnOfMonth", got)
		}
		if tom.entryDay != 24 || tom.exitAfter != 5 {
			t.Errorf("entryDay/exitAfter = %d/%d, want 24/5", tom.entryDay, tom.exitAfter)
		}
	})

	t.Run("partial map keeps defaults", func(t *testing.T) {
		got, err := base.WithParams(map[string]float64{"entryDay": 24})
		if err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		tom := got.(*turnOfMonth)
		if tom.entryDay != 24 {
			t.Errorf("entryDay = %d, want 24", tom.entryDay)
		}
		if tom.exitAfter != 3 {
			t.Errorf("exitAfter = %d, want default 3", tom.exitAfter)
		}
	})

	t.Run("unknown key rejected", func(t *testing.T) {
		if _, err := base.WithParams(map[string]float64{"bogus": 1}); err == nil {
			t.Fatal("WithParams() error = nil, want error for unknown key")
		}
	})

	t.Run("non-integral entryDay rejected", func(t *testing.T) {
		if _, err := base.WithParams(map[string]float64{"entryDay": 24.5}); err == nil {
			t.Fatal("WithParams() error = nil, want error for non-integral entryDay")
		}
	})

	t.Run("entryDay out of range rejected", func(t *testing.T) {
		if _, err := base.WithParams(map[string]float64{"entryDay": 0}); err == nil {
			t.Fatal("WithParams() error = nil, want error for entryDay<1")
		}
		if _, err := base.WithParams(map[string]float64{"entryDay": 32}); err == nil {
			t.Fatal("WithParams() error = nil, want error for entryDay>31")
		}
	})

	t.Run("exitAfter < 1 rejected", func(t *testing.T) {
		if _, err := base.WithParams(map[string]float64{"exitAfter": 0}); err == nil {
			t.Fatal("WithParams() error = nil, want error for exitAfter<1")
		}
	})

	t.Run("non-integral exitAfter rejected", func(t *testing.T) {
		if _, err := base.WithParams(map[string]float64{"exitAfter": 3.5}); err == nil {
			t.Fatal("WithParams() error = nil, want error for non-integral exitAfter")
		}
	})

	t.Run("receiver not mutated", func(t *testing.T) {
		if _, err := base.WithParams(map[string]float64{"entryDay": 24, "exitAfter": 5}); err != nil {
			t.Fatalf("WithParams() error = %v", err)
		}
		if base.entryDay != 26 || base.exitAfter != 3 {
			t.Errorf("receiver mutated: entryDay/exitAfter = %d/%d, want unchanged 26/3", base.entryDay, base.exitAfter)
		}
	})

	t.Run("ParamSpace/Grid has the documented 4 combos", func(t *testing.T) {
		defs := newTurnOfMonth().ParamSpace()
		combos := strategy.Grid(defs)
		if len(combos) != 4 {
			t.Fatalf("len(Grid(ParamSpace())) = %d, want 4", len(combos))
		}
	})
}

var _ strategy.Tunable = (*turnOfMonth)(nil)
var _ strategy.TargetWeighter = (*turnOfMonth)(nil)
