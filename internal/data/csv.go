package data

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"time"

	"tradeforge/internal/domain"
)

// csvDateLayout is the expected date format for the `date` column: YYYY-MM-DD.
const csvDateLayout = "2006-01-02"

// wantHeader is the required CSV header, in order.
var wantHeader = []string{"date", "open", "high", "low", "close", "volume"}

// LoadCSV reads a daily-bar CSV file for symbol and returns its bars in
// chronological order. The file must have header
// `date,open,high,low,close,volume` with dates in YYYY-MM-DD form (parsed as
// UTC midnight). Rows are validated: chronological order (strictly
// increasing dates), positive prices, and high >= low, high >= open, high >=
// close, low <= open, low <= close. The first validation failure produces a
// descriptive error naming the offending row.
func LoadCSV(path string, symbol string) ([]domain.Bar, error) {
	f, err := os.Open(path)
	if err != nil {
		// os.Open's *PathError already stringifies as "open <path>: <reason>",
		// so wrapping with another "open %s:" double-prints the path.
		return nil, fmt.Errorf("data: %w", err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.FieldsPerRecord = -1

	header, err := r.Read()
	if err != nil {
		return nil, fmt.Errorf("data: read header from %s: %w", path, err)
	}
	if !equalHeader(header, wantHeader) {
		return nil, fmt.Errorf("data: %s: expected header %v, got %v", path, wantHeader, header)
	}

	var bars []domain.Bar
	rowNum := 1 // header was row 1
	var prevTime time.Time
	for {
		rec, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("data: %s: row %d: %w", path, rowNum+1, err)
		}
		rowNum++

		if len(rec) != len(wantHeader) {
			return nil, fmt.Errorf("data: %s: row %d: expected %d fields, got %d", path, rowNum, len(wantHeader), len(rec))
		}

		bar, err := parseRow(rec)
		if err != nil {
			return nil, fmt.Errorf("data: %s: row %d: %w", path, rowNum, err)
		}
		bar.Symbol = symbol

		if err := validateBar(bar); err != nil {
			return nil, fmt.Errorf("data: %s: row %d (%s): %w", path, rowNum, bar.Time.Format(csvDateLayout), err)
		}

		if !prevTime.IsZero() && !bar.Time.After(prevTime) {
			return nil, fmt.Errorf("data: %s: row %d (%s): out of chronological order (previous %s)",
				path, rowNum, bar.Time.Format(csvDateLayout), prevTime.Format(csvDateLayout))
		}
		prevTime = bar.Time

		bars = append(bars, bar)
	}

	if len(bars) == 0 {
		return nil, fmt.Errorf("data: %s: no data rows", path)
	}

	return bars, nil
}

func equalHeader(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range want {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

func parseRow(rec []string) (domain.Bar, error) {
	var bar domain.Bar

	t, err := time.ParseInLocation(csvDateLayout, rec[0], time.UTC)
	if err != nil {
		return bar, fmt.Errorf("invalid date %q: %w", rec[0], err)
	}
	bar.Time = t

	open, err := strconv.ParseFloat(rec[1], 64)
	if err != nil {
		return bar, fmt.Errorf("invalid open %q: %w", rec[1], err)
	}
	bar.Open = open

	high, err := strconv.ParseFloat(rec[2], 64)
	if err != nil {
		return bar, fmt.Errorf("invalid high %q: %w", rec[2], err)
	}
	bar.High = high

	low, err := strconv.ParseFloat(rec[3], 64)
	if err != nil {
		return bar, fmt.Errorf("invalid low %q: %w", rec[3], err)
	}
	bar.Low = low

	closeP, err := strconv.ParseFloat(rec[4], 64)
	if err != nil {
		return bar, fmt.Errorf("invalid close %q: %w", rec[4], err)
	}
	bar.Close = closeP

	vol, err := strconv.ParseInt(rec[5], 10, 64)
	if err != nil {
		return bar, fmt.Errorf("invalid volume %q: %w", rec[5], err)
	}
	bar.Volume = vol

	return bar, nil
}

func validateBar(b domain.Bar) error {
	if b.Open <= 0 || b.High <= 0 || b.Low <= 0 || b.Close <= 0 {
		return fmt.Errorf("non-positive price (open=%.4f high=%.4f low=%.4f close=%.4f)", b.Open, b.High, b.Low, b.Close)
	}
	if b.High < b.Low {
		return fmt.Errorf("high (%.4f) < low (%.4f)", b.High, b.Low)
	}
	if b.High < b.Open || b.High < b.Close {
		return fmt.Errorf("high (%.4f) below open/close (open=%.4f close=%.4f)", b.High, b.Open, b.Close)
	}
	if b.Low > b.Open || b.Low > b.Close {
		return fmt.Errorf("low (%.4f) above open/close (open=%.4f close=%.4f)", b.Low, b.Open, b.Close)
	}
	if b.Volume < 0 {
		return fmt.Errorf("negative volume (%d)", b.Volume)
	}
	return nil
}
