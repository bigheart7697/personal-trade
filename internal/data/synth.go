package data

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"time"

	"tradeforge/internal/domain"
)

// regime is one alternating volatility/drift block of the synthetic price
// path. Mixing trending and mean-reverting-flavored regimes gives both
// trend-following and mean-reversion strategies something to react to.
type regime struct {
	days        int
	annualDrift float64 // e.g. 0.15 = +15%/yr
	annualVol   float64 // e.g. 0.20 = 20%/yr
}

const tradingDaysPerYear = 252

// Generate produces a deterministic slice of daily OHLCV bars for a single
// synthetic symbol ("SYNTH"), starting at startPrice, spanning approximately
// `years` years of weekdays-only trading days. The path is geometric
// Brownian motion driven by math/rand seeded with seed, walking through 3
// alternating regimes (trend-up/low-vol, choppy/high-vol, trend-down/mid-vol)
// so both trend-following and mean-reversion strategies get exercised.
// Bars are strictly chronological, Monday-Friday only, and O/H/L are derived
// plausibly around each day's close.
func Generate(seed int64, years int, startPrice float64) []domain.Bar {
	rng := newRand(seed)

	totalDays := years * tradingDaysPerYear
	if totalDays <= 0 {
		totalDays = tradingDaysPerYear
	}

	regimes := buildRegimes(totalDays)

	bars := make([]domain.Bar, 0, totalDays)
	price := startPrice
	if price <= 0 {
		price = 100
	}

	day := time.Date(2000, 1, 3, 0, 0, 0, 0, time.UTC) // a Monday
	for _, reg := range regimes {
		dailyDrift := reg.annualDrift / float64(tradingDaysPerYear)
		dailyVol := reg.annualVol / math.Sqrt(float64(tradingDaysPerYear))

		for i := 0; i < reg.days; i++ {
			// advance to next weekday
			for day.Weekday() == time.Saturday || day.Weekday() == time.Sunday {
				day = day.AddDate(0, 0, 1)
			}

			z := rng.NormFloat64()
			// GBM log-return step.
			logReturn := (dailyDrift - 0.5*dailyVol*dailyVol) + dailyVol*z
			open := price
			closeP := open * math.Exp(logReturn)
			if closeP <= 0 {
				closeP = open * 0.999 // guard against pathological collapse
			}

			high, low := deriveHighLow(rng, open, closeP)
			vol := int64(500000 + rng.Intn(2000000))

			bars = append(bars, domain.Bar{
				Symbol: "SYNTH",
				Time:   day,
				Open:   round2(open),
				High:   round2(high),
				Low:    round2(low),
				Close:  round2(closeP),
				Volume: vol,
			})

			price = closeP
			day = day.AddDate(0, 0, 1)
		}
	}

	return bars
}

// buildRegimes splits totalDays across three alternating regimes: a
// low-vol uptrend, a high-vol choppy/mean-reverting stretch, and a mid-vol
// downtrend, roughly in thirds (remainder folded into the last regime).
func buildRegimes(totalDays int) []regime {
	third := totalDays / 3
	last := totalDays - 2*third

	return []regime{
		{days: third, annualDrift: 0.15, annualVol: 0.12}, // calm uptrend
		{days: third, annualDrift: 0.0, annualVol: 0.35},  // choppy, mean-reverting flavor
		{days: last, annualDrift: -0.10, annualVol: 0.22}, // drawdown / downtrend
	}
}

// deriveHighLow builds a plausible high/low around an open/close pair using
// a small random intrabar range.
func deriveHighLow(rng randSource, open, closeP float64) (high, low float64) {
	base := math.Max(open, closeP)
	bottom := math.Min(open, closeP)
	// intrabar excursion: 0 to ~1.5% beyond the open/close range on each side
	upExcursion := base * (0.001 + 0.015*rng.Float64())
	downExcursion := bottom * (0.001 + 0.015*rng.Float64())
	high = base + upExcursion
	low = bottom - downExcursion
	if low <= 0 {
		low = bottom * 0.999
	}
	return high, low
}

func round2(f float64) float64 {
	return math.Round(f*100) / 100
}

// WriteCSV writes bars to path in the `date,open,high,low,close,volume`
// format expected by LoadCSV.
func WriteCSV(path string, bars []domain.Bar) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("data: create %s: %w", path, err)
	}
	defer f.Close()

	w := csv.NewWriter(f)
	if err := w.Write(wantHeader); err != nil {
		return fmt.Errorf("data: write header to %s: %w", path, err)
	}

	for _, b := range bars {
		rec := []string{
			b.Time.Format(csvDateLayout),
			fmt.Sprintf("%.2f", b.Open),
			fmt.Sprintf("%.2f", b.High),
			fmt.Sprintf("%.2f", b.Low),
			fmt.Sprintf("%.2f", b.Close),
			fmt.Sprintf("%d", b.Volume),
		}
		if err := w.Write(rec); err != nil {
			return fmt.Errorf("data: write row to %s: %w", path, err)
		}
	}

	w.Flush()
	if err := w.Error(); err != nil {
		return fmt.Errorf("data: flush %s: %w", path, err)
	}
	return nil
}
