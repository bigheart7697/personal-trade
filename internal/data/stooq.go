package data

import (
	"context"
	"encoding/csv"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"tradeforge/internal/domain"
)

// maxStooqBodyBytes caps how much of a Stooq response we will read, guarding
// against an unexpectedly huge or endless body (e.g. a misbehaving proxy).
const maxStooqBodyBytes = 20 * 1024 * 1024 // 20 MB

// stooqHeaderWithVolume and stooqHeaderNoVolume are the two header shapes
// Stooq's daily CSV export is known to return; indices (e.g. ^spx) sometimes
// omit the Volume column entirely.
var (
	stooqHeaderWithVolume = []string{"date", "open", "high", "low", "close", "volume"}
	stooqHeaderNoVolume   = []string{"date", "open", "high", "low", "close"}
)

// StooqClient fetches free daily EOD bars from Stooq
// (https://stooq.com/q/d/l/?s=<symbol>&i=d). It satisfies data.BarSource.
type StooqClient struct {
	HTTPClient *http.Client
	BaseURL    string
}

// NewStooqClient returns a StooqClient with sane defaults: a 30s timeout and
// Stooq's daily-download endpoint.
func NewStooqClient() *StooqClient {
	return &StooqClient{
		HTTPClient: &http.Client{Timeout: 30 * time.Second},
		BaseURL:    "https://stooq.com/q/d/l/",
	}
}

// Bars fetches and parses the daily bar history for symbol from Stooq. The
// input symbol is a plain ticker (e.g. "SPY") or an index (e.g. "^SPX");
// stooqSymbol maps it to the form Stooq expects. The returned bars carry the
// input symbol, uppercased, regardless of the form Stooq required in the
// request.
func (c *StooqClient) Bars(ctx context.Context, symbol string) ([]domain.Bar, error) {
	httpClient := c.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 30 * time.Second}
	}
	baseURL := c.BaseURL
	if baseURL == "" {
		baseURL = "https://stooq.com/q/d/l/"
	}

	mapped := stooqSymbol(symbol)
	outSymbol := strings.ToUpper(symbol)

	q := url.Values{}
	q.Set("s", mapped)
	q.Set("i", "d")
	reqURL := baseURL + "?" + q.Encode()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if err != nil {
		return nil, fmt.Errorf("data: stooq: build request for %s: %w", symbol, err)
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("data: stooq: fetch %s: %w", symbol, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("data: stooq: fetch %s: unexpected status %d %s", symbol, resp.StatusCode, resp.Status)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxStooqBodyBytes))
	if err != nil {
		return nil, fmt.Errorf("data: stooq: read response for %s: %w", symbol, err)
	}

	trimmed := strings.TrimSpace(string(body))
	if strings.HasPrefix(trimmed, "<") {
		return nil, fmt.Errorf("data: stooq: stooq returned an HTML page instead of CSV for %s "+
			"(browser verification or daily download limit); download the CSV manually from "+
			"https://stooq.com/q/d/?s=%s and run: tradeforge data import --in <file> --symbol %s --out data/%s.csv",
			symbol, mapped, outSymbol, outSymbol)
	}
	if strings.HasPrefix(strings.ToLower(trimmed), "no data") {
		return nil, fmt.Errorf("data: stooq: no data exists for symbol %s", symbol)
	}

	bars, err := parseStooqCSV(strings.NewReader(trimmed), outSymbol)
	if err != nil {
		return nil, fmt.Errorf("data: stooq: %s: %w", symbol, err)
	}

	return bars, nil
}

// LoadStooqCSV parses a Stooq-format daily CSV file the user downloaded by
// hand from stooq.com in their browser (the endpoint currently fronts its
// programmatic download with a JavaScript verification challenge, so a
// browser download is the reliable path). Accepts either the
// Date,Open,High,Low,Close,Volume header or the no-Volume variant. Returned
// bars carry symbol, uppercased.
func LoadStooqCSV(path string, symbol string) ([]domain.Bar, error) {
	f, err := os.Open(path)
	if err != nil {
		// A missing LOCAL input file is not a Stooq-format failure, so it must
		// not carry the "stooq:" label (that reads as if a network fetch
		// failed). The *PathError already names the path.
		return nil, fmt.Errorf("data: %w", err)
	}
	defer f.Close()

	bars, err := parseStooqCSV(f, strings.ToUpper(symbol))
	if err != nil {
		return nil, fmt.Errorf("data: stooq: %s: %w", path, err)
	}

	return bars, nil
}

// stooqSymbol maps an input ticker to the form Stooq's endpoint expects:
// symbols already containing '.' (e.g. "spy.us") or '^' (index prefix, e.g.
// "^spx") are only lowercased; plain tickers are lowercased and given the
// ".us" suffix.
func stooqSymbol(symbol string) string {
	lower := strings.ToLower(symbol)
	if strings.ContainsAny(lower, ".^") {
		return lower
	}
	return lower + ".us"
}

// parseStooqCSV parses a Stooq daily CSV body (with or without a Volume
// column) into chronologically ordered, validated bars labeled with symbol.
func parseStooqCSV(r io.Reader, symbol string) ([]domain.Bar, error) {
	cr := csv.NewReader(r)
	cr.FieldsPerRecord = -1

	header, err := cr.Read()
	if err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}

	hasVolume, err := matchStooqHeader(header)
	if err != nil {
		return nil, err
	}

	var bars []domain.Bar
	rowNum := 1 // header was row 1
	var prevTime time.Time
	for {
		rec, err := cr.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("row %d: %w", rowNum+1, err)
		}
		rowNum++

		wantFields := len(stooqHeaderWithVolume)
		if !hasVolume {
			wantFields = len(stooqHeaderNoVolume)
		}
		if len(rec) != wantFields {
			return nil, fmt.Errorf("row %d: expected %d fields, got %d", rowNum, wantFields, len(rec))
		}

		bar, err := parseStooqRow(rec, hasVolume)
		if err != nil {
			return nil, fmt.Errorf("row %d: %w", rowNum, err)
		}
		bar.Symbol = symbol

		if err := validateBar(bar); err != nil {
			return nil, fmt.Errorf("row %d (%s): %w", rowNum, bar.Time.Format(csvDateLayout), err)
		}

		if !prevTime.IsZero() && !bar.Time.After(prevTime) {
			return nil, fmt.Errorf("row %d (%s): out of chronological order (previous %s)",
				rowNum, bar.Time.Format(csvDateLayout), prevTime.Format(csvDateLayout))
		}
		prevTime = bar.Time

		bars = append(bars, bar)
	}

	if len(bars) == 0 {
		return nil, fmt.Errorf("no data rows")
	}

	return bars, nil
}

// matchStooqHeader checks header against the two accepted Stooq shapes
// case-insensitively and reports whether a Volume column is present.
func matchStooqHeader(header []string) (hasVolume bool, err error) {
	if equalHeaderFold(header, stooqHeaderWithVolume) {
		return true, nil
	}
	if equalHeaderFold(header, stooqHeaderNoVolume) {
		return false, nil
	}
	return false, fmt.Errorf("unexpected header %v", header)
}

func equalHeaderFold(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range want {
		if !strings.EqualFold(strings.TrimSpace(got[i]), want[i]) {
			return false
		}
	}
	return true
}

func parseStooqRow(rec []string, hasVolume bool) (domain.Bar, error) {
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

	if hasVolume {
		volField := strings.TrimSpace(rec[5])
		if volField == "" || volField == "-" {
			bar.Volume = 0
		} else {
			vol, err := strconv.ParseInt(volField, 10, 64)
			if err != nil {
				return bar, fmt.Errorf("invalid volume %q: %w", rec[5], err)
			}
			bar.Volume = vol
		}
	}

	return bar, nil
}
