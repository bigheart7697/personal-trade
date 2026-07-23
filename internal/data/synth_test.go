package data

import (
	"testing"
	"time"
)

func TestGenerate_Determinism(t *testing.T) {
	a := Generate(42, 2, 100)
	b := Generate(42, 2, 100)

	if len(a) != len(b) {
		t.Fatalf("length mismatch: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("bar %d differs: %+v vs %+v", i, a[i], b[i])
		}
	}
}

func TestGenerate_DifferentSeedsDiffer(t *testing.T) {
	a := Generate(1, 2, 100)
	b := Generate(2, 2, 100)

	if len(a) != len(b) {
		t.Fatalf("length mismatch: %d vs %d", len(a), len(b))
	}
	same := true
	for i := range a {
		if a[i].Close != b[i].Close {
			same = false
			break
		}
	}
	if same {
		t.Error("different seeds produced identical close prices")
	}
}

func TestGenerate_WeekdaysOnly(t *testing.T) {
	bars := Generate(7, 1, 100)
	for _, b := range bars {
		wd := b.Time.Weekday()
		if wd == time.Saturday || wd == time.Sunday {
			t.Fatalf("bar on %s at %s (weekday), expected weekdays only", wd, b.Time.Format("2006-01-02"))
		}
	}
}

func TestGenerate_ChronologicalOrder(t *testing.T) {
	bars := Generate(3, 2, 100)
	for i := 1; i < len(bars); i++ {
		if !bars[i].Time.After(bars[i-1].Time) {
			t.Fatalf("bar %d (%s) not strictly after bar %d (%s)",
				i, bars[i].Time, i-1, bars[i-1].Time)
		}
	}
}

func TestGenerate_OHLCSanity(t *testing.T) {
	bars := Generate(99, 3, 100)
	if len(bars) == 0 {
		t.Fatal("Generate() produced no bars")
	}
	for i, b := range bars {
		if b.Open <= 0 || b.High <= 0 || b.Low <= 0 || b.Close <= 0 {
			t.Fatalf("bar %d has non-positive price: %+v", i, b)
		}
		if b.High < b.Low {
			t.Fatalf("bar %d: high (%v) < low (%v)", i, b.High, b.Low)
		}
		if b.High < b.Open || b.High < b.Close {
			t.Fatalf("bar %d: high (%v) below open/close (open=%v close=%v)", i, b.High, b.Open, b.Close)
		}
		if b.Low > b.Open || b.Low > b.Close {
			t.Fatalf("bar %d: low (%v) above open/close (open=%v close=%v)", i, b.Low, b.Open, b.Close)
		}
		if b.Volume <= 0 {
			t.Fatalf("bar %d has non-positive volume: %v", i, b.Volume)
		}
		if b.Symbol == "" {
			t.Fatalf("bar %d has empty symbol", i)
		}
	}
}

func TestGenerate_UTCTimes(t *testing.T) {
	bars := Generate(5, 1, 100)
	for i, b := range bars {
		if b.Time.Location() != time.UTC {
			t.Fatalf("bar %d time location = %v, want UTC", i, b.Time.Location())
		}
	}
}

func TestGenerate_ApproximateLength(t *testing.T) {
	years := 5
	bars := Generate(1, years, 100)
	wantApprox := years * tradingDaysPerYear
	// Allow generous tolerance since regime day-splitting can round.
	if bars == nil || len(bars) < wantApprox-10 || len(bars) > wantApprox+10 {
		t.Errorf("len(bars) = %d, want approximately %d", len(bars), wantApprox)
	}
}

func TestGenerate_RoundTripCSV(t *testing.T) {
	dir := t.TempDir()
	path := dir + "/synth.csv"

	bars := Generate(42, 1, 50)
	if err := WriteCSV(path, bars); err != nil {
		t.Fatalf("WriteCSV() error = %v", err)
	}

	loaded, err := LoadCSV(path, "SYNTH")
	if err != nil {
		t.Fatalf("LoadCSV() error = %v", err)
	}

	if len(loaded) != len(bars) {
		t.Fatalf("round-trip length mismatch: wrote %d, loaded %d", len(bars), len(loaded))
	}
	for i := range bars {
		if !loaded[i].Time.Equal(bars[i].Time) {
			t.Errorf("bar %d time mismatch: wrote %v, loaded %v", i, bars[i].Time, loaded[i].Time)
		}
		if loaded[i].Close != bars[i].Close {
			t.Errorf("bar %d close mismatch: wrote %v, loaded %v", i, bars[i].Close, loaded[i].Close)
		}
	}
}
