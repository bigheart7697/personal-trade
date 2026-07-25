package data

import (
	"strings"
	"testing"
	"time"

	"tradeforge/internal/domain"
)

func mkQualityBar(dateStr string, closeP float64, volume int64) domain.Bar {
	t, err := time.Parse(csvDateLayout, dateStr)
	if err != nil {
		panic(err)
	}
	return domain.Bar{
		Symbol: "TEST",
		Time:   t,
		Open:   closeP,
		High:   closeP * 1.01,
		Low:    closeP * 0.99,
		Close:  closeP,
		Volume: volume,
	}
}

func TestCheckQuality_CleanWeekendsNotFlagged(t *testing.T) {
	// Fri -> Mon is a normal 2-day weekend gap: 0 missing weekdays.
	bars := []domain.Bar{
		mkQualityBar("2021-01-01", 100, 1000), // Friday
		mkQualityBar("2021-01-04", 101, 1000), // Monday
		mkQualityBar("2021-01-05", 102, 1000), // Tuesday
	}

	report := CheckQuality("TEST", bars)

	if len(report.Gaps) != 0 {
		t.Errorf("Gaps = %+v, want none (weekend only)", report.Gaps)
	}
	if !report.Clean() {
		t.Errorf("Clean() = false, want true")
	}
}

func TestCheckQuality_GapFlagged(t *testing.T) {
	// Fri 2021-01-01 -> following Fri 2021-01-08 skips a full week:
	// Mon 1/4, Tue 1/5, Wed 1/6, Thu 1/7 = 4 missing weekdays.
	bars := []domain.Bar{
		mkQualityBar("2021-01-01", 100, 1000),
		mkQualityBar("2021-01-08", 101, 1000),
	}

	report := CheckQuality("TEST", bars)

	if len(report.Gaps) != 1 {
		t.Fatalf("len(Gaps) = %d, want 1", len(report.Gaps))
	}
	g := report.Gaps[0]
	if g.MissingWeekdays != 4 {
		t.Errorf("MissingWeekdays = %d, want 4", g.MissingWeekdays)
	}
	wantFrom, _ := time.Parse(csvDateLayout, "2021-01-01")
	wantTo, _ := time.Parse(csvDateLayout, "2021-01-08")
	if !g.From.Equal(wantFrom) {
		t.Errorf("From = %v, want %v", g.From, wantFrom)
	}
	if !g.To.Equal(wantTo) {
		t.Errorf("To = %v, want %v", g.To, wantTo)
	}
	if report.Clean() {
		t.Errorf("Clean() = true, want false (gap present)")
	}
}

func TestCheckQuality_SuspectMoves(t *testing.T) {
	bars := []domain.Bar{
		mkQualityBar("2021-01-01", 100, 1000),
		mkQualityBar("2021-01-04", 130, 1000), // +30%
		mkQualityBar("2021-01-05", 91, 1000),  // -30% from 130
	}

	report := CheckQuality("TEST", bars)

	if len(report.SuspectMoves) != 2 {
		t.Fatalf("len(SuspectMoves) = %d, want 2 (%+v)", len(report.SuspectMoves), report.SuspectMoves)
	}

	up := report.SuspectMoves[0]
	if up.ChangePct <= 0 {
		t.Errorf("first suspect move ChangePct = %v, want positive", up.ChangePct)
	}
	if up.PrevClose != 100 || up.Close != 130 {
		t.Errorf("first suspect move = %+v, want PrevClose=100 Close=130", up)
	}

	down := report.SuspectMoves[1]
	if down.ChangePct >= 0 {
		t.Errorf("second suspect move ChangePct = %v, want negative", down.ChangePct)
	}
	if down.PrevClose != 130 || down.Close != 91 {
		t.Errorf("second suspect move = %+v, want PrevClose=130 Close=91", down)
	}

	if report.Clean() {
		t.Errorf("Clean() = true, want false (suspect moves present)")
	}
}

func TestCheckQuality_ZeroVolumeCounted(t *testing.T) {
	bars := []domain.Bar{
		mkQualityBar("2021-01-01", 100, 0),
		mkQualityBar("2021-01-04", 101, 1000),
		mkQualityBar("2021-01-05", 102, 0),
	}

	report := CheckQuality("TEST", bars)

	if report.ZeroVolumeBars != 2 {
		t.Errorf("ZeroVolumeBars = %d, want 2", report.ZeroVolumeBars)
	}
	// Zero-volume bars alone must not fail Clean().
	if !report.Clean() {
		t.Errorf("Clean() = false, want true (zero volume alone should not fail)")
	}
}

func TestCheckQuality_Empty(t *testing.T) {
	report := CheckQuality("TEST", nil)

	if report.Symbol != "TEST" {
		t.Errorf("Symbol = %q, want TEST", report.Symbol)
	}
	if report.NumBars != 0 {
		t.Errorf("NumBars = %d, want 0", report.NumBars)
	}
	if len(report.Gaps) != 0 || len(report.SuspectMoves) != 0 {
		t.Errorf("expected no gaps/suspect moves for empty input, got %+v", report)
	}
	if !report.Clean() {
		t.Errorf("Clean() = false, want true for empty input")
	}
}

func TestQualityReport_String(t *testing.T) {
	t.Run("clean says so explicitly", func(t *testing.T) {
		bars := []domain.Bar{
			mkQualityBar("2021-01-01", 100, 1000),
			mkQualityBar("2021-01-04", 101, 1000),
		}
		report := CheckQuality("TEST", bars)
		s := report.String()
		if !strings.Contains(s, "Clean") && !strings.Contains(s, "clean") {
			t.Errorf("String() = %q, want it to mention clean status", s)
		}
	})

	t.Run("caps printed items at 10 per category", func(t *testing.T) {
		day := time.Date(2021, 1, 1, 0, 0, 0, 0, time.UTC)
		bars := []domain.Bar{{
			Symbol: "TEST", Time: day, Open: 100, High: 101, Low: 99, Close: 100, Volume: 1000,
		}}
		for i := 0; i < 15; i++ {
			day = day.AddDate(0, 0, 14) // skip 2 weeks -> guaranteed gap each time
			bars = append(bars, domain.Bar{
				Symbol: "TEST",
				Time:   day,
				Open:   100,
				High:   101,
				Low:    99,
				Close:  100,
				Volume: 1000,
			})
		}

		report := CheckQuality("TEST", bars)
		if len(report.Gaps) != 15 {
			t.Fatalf("len(Gaps) = %d, want 15", len(report.Gaps))
		}
		s := report.String()
		if !strings.Contains(s, "and 5 more") {
			t.Errorf("String() = %q, want it to cap at 10 and mention 5 more", s)
		}
	})
}
