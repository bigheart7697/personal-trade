package data

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"tradeforge/internal/domain"
)

// RowProblem is one structural defect found at a specific 1-based file row
// (the header is row 1) while inspecting a CSV. Detail is a human-readable
// description (bad field count, unparseable value, invalid OHLC, out of
// chronological order).
type RowProblem struct {
	Row    int
	Detail string
}

// Inspection is the full survey of a CSV's health, produced by InspectCSV.
// Unlike LoadCSV — which fails fast on the first structural error because it
// feeds the backtester and must never hand a strategy a malformed bar —
// InspectCSV keeps going and collects EVERY problem, so `data check` can
// diagnose a messy file in one pass. Bars holds only the rows that parsed
// and validated cleanly, so Quality is a genuine quality scan over the good
// data even when some rows were rejected.
type Inspection struct {
	Path          string
	Symbol        string
	HeaderProblem string // "" when the header matched the canonical layout
	RowProblems   []RowProblem
	Bars          []domain.Bar
	Quality       QualityReport
}

// OK reports whether the file is fully healthy: canonical header, no
// structural row problems, and a clean quality report. Zero-volume bars do
// not fail OK (see QualityReport.Clean) — they are legitimate for many free
// data sources.
func (in Inspection) OK() bool {
	return in.HeaderProblem == "" && len(in.RowProblems) == 0 && in.Quality.Clean()
}

// InspectCSV surveys a daily-bar CSV without failing fast: it records a
// header mismatch, every unparseable/invalid/out-of-order row, and then runs
// the quality scan (gaps, suspect moves, zero-volume) over whatever rows
// parsed cleanly. It only returns an error when the file itself cannot be
// opened or read as CSV at all; individual bad rows become RowProblems, not
// a returned error. Column parsing is positional (canonical order), so rows
// are still surveyed even when the header is wrong — the mismatch is reported
// separately.
func InspectCSV(path, symbol string) (Inspection, error) {
	in := Inspection{Path: path, Symbol: symbol}

	f, err := os.Open(path)
	if err != nil {
		return in, fmt.Errorf("data: %w", err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.FieldsPerRecord = -1

	header, err := r.Read()
	if err != nil {
		return in, fmt.Errorf("data: read header from %s: %w", path, err)
	}
	if !equalHeader(header, wantHeader) {
		in.HeaderProblem = fmt.Sprintf("expected header %v, got %v", wantHeader, header)
	}

	rowNum := 1 // header was row 1
	var prevTime time.Time
	for {
		rec, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			// A malformed CSV record (e.g. a bad quote) — record it and keep
			// going; the csv reader recovers on the next line.
			rowNum++
			in.RowProblems = append(in.RowProblems, RowProblem{Row: rowNum, Detail: err.Error()})
			continue
		}
		rowNum++

		if len(rec) != len(wantHeader) {
			in.RowProblems = append(in.RowProblems, RowProblem{
				Row:    rowNum,
				Detail: fmt.Sprintf("expected %d fields, got %d", len(wantHeader), len(rec)),
			})
			continue
		}

		bar, err := parseRow(rec)
		if err != nil {
			in.RowProblems = append(in.RowProblems, RowProblem{Row: rowNum, Detail: err.Error()})
			continue
		}
		bar.Symbol = symbol

		if err := validateBar(bar); err != nil {
			in.RowProblems = append(in.RowProblems, RowProblem{
				Row:    rowNum,
				Detail: fmt.Sprintf("%s: %v", bar.Time.Format(csvDateLayout), err),
			})
			continue
		}

		if !prevTime.IsZero() && !bar.Time.After(prevTime) {
			in.RowProblems = append(in.RowProblems, RowProblem{
				Row:    rowNum,
				Detail: fmt.Sprintf("%s: out of chronological order (previous %s)", bar.Time.Format(csvDateLayout), prevTime.Format(csvDateLayout)),
			})
			// Do not advance prevTime or keep this bar: the quality scan
			// assumes strictly increasing times.
			continue
		}
		prevTime = bar.Time
		in.Bars = append(in.Bars, bar)
	}

	in.Quality = CheckQuality(symbol, in.Bars)
	return in, nil
}

// String renders a compact, human-readable inspection: the header problem
// (if any), each structural row problem (capped like the quality report),
// then the full quality report over the clean rows.
func (in Inspection) String() string {
	var b strings.Builder

	fmt.Fprintf(&b, "Inspecting: %s (%s)\n", in.Path, in.Symbol)

	if in.HeaderProblem != "" {
		fmt.Fprintf(&b, "  Header problem: %s\n", in.HeaderProblem)
	}

	if len(in.RowProblems) == 0 {
		fmt.Fprintf(&b, "  Structural rows: all valid (%d parsed)\n", len(in.Bars))
	} else {
		fmt.Fprintf(&b, "  Structural problems (%d):\n", len(in.RowProblems))
		for i, p := range in.RowProblems {
			if i >= maxPrintedIssues {
				fmt.Fprintf(&b, "    ... and %d more\n", len(in.RowProblems)-maxPrintedIssues)
				break
			}
			fmt.Fprintf(&b, "    row %d: %s\n", p.Row, p.Detail)
		}
	}

	b.WriteString(in.Quality.String())
	return b.String()
}
