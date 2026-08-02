package eval

import (
	"math"
	"testing"
	"time"

	"tradeforge/internal/domain"
	"tradeforge/internal/metrics"
)

// linearBars returns n daily bars (weekdays only) whose close moves by a
// constant additive step per bar, starting at startPrice. A positive step
// produces a steadily rising series; a negative step, steadily falling.
// Deterministic and dependency-free, like risingBars but with a
// caller-controlled direction and additive (not multiplicative) step so a
// down-leg can be built simply by negating the step.
func linearBars(n int, startPrice, step float64) []domain.Bar {
	bars := make([]domain.Bar, 0, n)
	day := time.Date(2000, 1, 3, 0, 0, 0, 0, time.UTC) // a Monday
	price := startPrice
	for i := 0; i < n; i++ {
		for day.Weekday() == time.Saturday || day.Weekday() == time.Sunday {
			day = day.AddDate(0, 0, 1)
		}
		open := price
		closeP := price + step
		high := math.Max(open, closeP)
		low := math.Min(open, closeP)
		bars = append(bars, domain.Bar{
			Symbol: "TEST",
			Time:   day,
			Open:   open,
			High:   high,
			Low:    low,
			Close:  closeP,
			Volume: 100000,
		})
		price = closeP
		day = day.AddDate(0, 0, 1)
	}
	return bars
}

func TestClassifyRegimes_BoundaryUnclassified(t *testing.T) {
	bars := linearBars(220, 100, 0.1)
	regimes, classified := ClassifyRegimes(bars)

	if len(regimes) != len(bars) || len(classified) != len(bars) {
		t.Fatalf("len(regimes)=%d len(classified)=%d, want %d", len(regimes), len(classified), len(bars))
	}

	// minClassifiedIdx is 220; with exactly 220 bars (indices 0..219), no
	// index reaches 220, so every bar is unclassified.
	for i := 0; i < len(bars); i++ {
		if classified[i] {
			t.Errorf("classified[%d] = true, want false (need index >= %d)", i, minClassifiedIdx)
		}
	}
}

func TestClassifyRegimes_RisingSeriesIsBull(t *testing.T) {
	// 400 rising bars: by the time the SMA/slope windows are both full and
	// have had time to actually reflect the trend, the regime should read
	// BULL consistently.
	bars := linearBars(400, 100, 0.5)
	regimes, classified := ClassifyRegimes(bars)

	// Check the tail of the series, well past both warmup windows.
	for i := 300; i < len(bars); i++ {
		if !classified[i] {
			t.Fatalf("classified[%d] = false, want true", i)
		}
		if regimes[i] != Bull {
			t.Errorf("regimes[%d] = %v, want Bull", i, regimes[i])
		}
	}
}

func TestClassifyRegimes_FallingSeriesIsBear(t *testing.T) {
	bars := linearBars(400, 500, -0.5)
	regimes, classified := ClassifyRegimes(bars)

	for i := 300; i < len(bars); i++ {
		if !classified[i] {
			t.Fatalf("classified[%d] = false, want true", i)
		}
		if regimes[i] != Bear {
			t.Errorf("regimes[%d] = %v, want Bear", i, regimes[i])
		}
	}
}

// TestClassifyRegimes_UpThenDownFlips builds a 400-bar up-leg followed by a
// 400-bar down-leg and asserts the classification flips from Bull to Bear
// (via Chop while the SMA catches up) as the series' trend reverses.
func TestClassifyRegimes_UpThenDownFlips(t *testing.T) {
	up := linearBars(400, 100, 0.5)
	lastPrice := up[len(up)-1].Close
	down := linearBars(400, lastPrice, -0.5)
	// down's own timestamps restart from day 0; shift them to continue
	// immediately after up's last bar (skipping weekends), and keep OHLC.
	day := up[len(up)-1].Time.AddDate(0, 0, 1)
	for i := range down {
		for day.Weekday() == time.Saturday || day.Weekday() == time.Sunday {
			day = day.AddDate(0, 0, 1)
		}
		down[i].Time = day
		day = day.AddDate(0, 0, 1)
	}
	bars := append(up, down...)

	regimes, classified := ClassifyRegimes(bars)

	// Well into the up-leg (before the down-leg starts influencing the
	// 200-bar SMA window), classification should read Bull.
	if !classified[399] || regimes[399] != Bull {
		t.Errorf("regimes[399] = %v (classified=%v), want Bull", regimes[399], classified[399])
	}

	// Well into the down-leg, once the SMA window is dominated by falling
	// prices and its slope has had time to turn down, classification should
	// read Bear.
	last := len(bars) - 1
	if !classified[last] || regimes[last] != Bear {
		t.Errorf("regimes[%d] = %v (classified=%v), want Bear", last, regimes[last], classified[last])
	}
}

func TestRegimeBreakdown_SharesAndCounts(t *testing.T) {
	bars := linearBars(400, 100, 0.5)

	// Build a synthetic OOS equity curve over the tail of bars (well past
	// classification warmup), tracking the underlying exactly so the strategy
	// slice and benchmark slice should look similar (not asserted exactly,
	// just used to exercise the accounting).
	oosStart := 350
	oosBars := bars[oosStart:]
	equity := make([]metrics.EquityPoint, len(oosBars))
	cash := 100000.0
	equity[0] = metrics.EquityPoint{Time: oosBars[0].Time, Equity: cash}
	for i := 1; i < len(oosBars); i++ {
		ret := oosBars[i].Close/oosBars[i-1].Close - 1
		cash *= 1 + ret
		equity[i] = metrics.EquityPoint{Time: oosBars[i].Time, Equity: cash}
	}

	slices, dropped := RegimeBreakdown(bars, equity)

	if len(slices) != 3 {
		t.Fatalf("len(slices) = %d, want 3", len(slices))
	}
	wantOrder := []Regime{Bull, Bear, Chop}
	for i, s := range slices {
		if s.Regime != wantOrder[i] {
			t.Errorf("slices[%d].Regime = %v, want %v", i, s.Regime, wantOrder[i])
		}
	}

	numReturns := len(metrics.DailyReturns(equity))
	daysSum := 0
	for _, s := range slices {
		daysSum += s.Days
	}
	if daysSum+dropped != numReturns {
		t.Errorf("daysSum(%d) + dropped(%d) = %d, want %d (len(oos returns))", daysSum, dropped, daysSum+dropped, numReturns)
	}

	// Since this OOS window is entirely inside the well-established up-leg,
	// all classified days should land in Bull, with zero-day Bear/Chop
	// slices present in the output.
	nonEmpty := 0
	shareSum := 0.0
	for _, s := range slices {
		if s.Days > 0 {
			nonEmpty++
			shareSum += s.Share
		}
	}
	if nonEmpty == 0 {
		t.Fatal("expected at least one non-empty regime slice")
	}
	const tol = 1e-9
	if diff := shareSum - 1.0; diff > tol || diff < -tol {
		t.Errorf("sum of non-empty Share = %v, want 1.0 (within %v)", shareSum, tol)
	}

	foundZero := false
	for _, s := range slices {
		if s.Days == 0 {
			foundZero = true
			if s.Share != 0 || s.StratAnnRet != 0 || s.StratSharpe != 0 || s.StratWorst != 0 ||
				s.BenchAnnRet != 0 || s.BenchSharpe != 0 || s.BenchWorst != 0 {
				t.Errorf("zero-day slice %v has non-zero stats: %+v", s.Regime, s)
			}
		}
	}
	if !foundZero {
		t.Error("expected at least one zero-day regime slice (Bear or Chop) in a purely rising OOS window")
	}

	// No NaN/Inf anywhere.
	for _, s := range slices {
		for _, f := range []float64{s.Share, s.StratAnnRet, s.StratSharpe, s.StratWorst, s.BenchAnnRet, s.BenchSharpe, s.BenchWorst} {
			if math.IsNaN(f) || math.IsInf(f, 0) {
				t.Errorf("slice %v has NaN/Inf field: %+v", s.Regime, s)
			}
		}
	}
}

func TestRegimeBreakdown_EmptyOOSEquity(t *testing.T) {
	bars := linearBars(400, 100, 0.5)

	slices, dropped := RegimeBreakdown(bars, nil)
	if len(slices) != 3 {
		t.Fatalf("len(slices) = %d, want 3", len(slices))
	}
	if dropped != 0 {
		t.Errorf("dropped = %d, want 0", dropped)
	}
	for _, s := range slices {
		if s.Days != 0 {
			t.Errorf("slice %v: Days = %d, want 0 for empty OOS equity", s.Regime, s.Days)
		}
	}
}

func TestRegimeBreakdown_UnknownTimeIsDropped(t *testing.T) {
	bars := linearBars(400, 100, 0.5)

	// Build an OOS curve whose second point's time does not exist in bars at
	// all, exercising the defensive "time not found -> drop" path.
	equity := []metrics.EquityPoint{
		{Time: bars[350].Time, Equity: 100000},
		{Time: time.Date(1999, 1, 1, 0, 0, 0, 0, time.UTC), Equity: 100500},
	}

	slices, dropped := RegimeBreakdown(bars, equity)
	if dropped != 1 {
		t.Errorf("dropped = %d, want 1", dropped)
	}
	total := 0
	for _, s := range slices {
		total += s.Days
	}
	if total != 0 {
		t.Errorf("total Days = %d, want 0 (the only return day was dropped)", total)
	}
}
