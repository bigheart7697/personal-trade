package data

import (
	"os"
	"path/filepath"
	"testing"
)

func writeInspectCSV(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "in.csv")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	return path
}

// TestInspectCSV_AggregatesEveryProblem is the whole point of the lenient
// inspector: unlike LoadCSV (fail-fast, feeds the backtester), InspectCSV
// keeps going and reports every structural defect in one pass, plus a quality
// scan over the rows that did parse. A backtest-fed loader can never diagnose
// the messy files `data check` exists for.
func TestInspectCSV_AggregatesEveryProblem(t *testing.T) {
	// Row 3 has high<low; row 5 has a non-numeric close; the rest are valid
	// with a large (>25%) close-to-close jump between rows 6 and 7.
	body := "date,open,high,low,close,volume\n" +
		"2020-01-02,100,101,99,100,1000\n" + // row 2 ok
		"2020-01-03,100,90,110,100,1000\n" + // row 3 high<low
		"2020-01-06,100,101,99,100,1000\n" + // row 4 ok
		"2020-01-07,100,101,99,abc,1000\n" + // row 5 bad close
		"2020-01-08,100,101,99,100,1000\n" + // row 6 ok
		"2020-01-09,100,205,99,200,1000\n" //   row 7 ok but +100% jump (suspect)

	insp, err := InspectCSV(writeInspectCSV(t, body), "TEST")
	if err != nil {
		t.Fatalf("InspectCSV error = %v", err)
	}
	if insp.HeaderProblem != "" {
		t.Errorf("HeaderProblem = %q, want empty", insp.HeaderProblem)
	}
	if len(insp.RowProblems) != 2 {
		t.Fatalf("RowProblems = %+v, want exactly 2 (rows 3 and 5)", insp.RowProblems)
	}
	if insp.RowProblems[0].Row != 3 || insp.RowProblems[1].Row != 5 {
		t.Errorf("problem rows = %d,%d, want 3,5", insp.RowProblems[0].Row, insp.RowProblems[1].Row)
	}
	// Four rows parsed cleanly (2,4,6,7); the quality scan runs over them and
	// must flag the +100% move as suspect.
	if len(insp.Bars) != 4 {
		t.Errorf("clean bars = %d, want 4", len(insp.Bars))
	}
	if len(insp.Quality.SuspectMoves) != 1 {
		t.Errorf("SuspectMoves = %+v, want 1", insp.Quality.SuspectMoves)
	}
	if insp.OK() {
		t.Error("OK() = true, want false (has structural problems and a suspect move)")
	}
}

// TestInspectCSV_CleanFileIsOK: a fully valid file has no problems and OK().
func TestInspectCSV_CleanFileIsOK(t *testing.T) {
	body := "date,open,high,low,close,volume\n" +
		"2020-01-02,100,101,99,100,1000\n" +
		"2020-01-03,100,101,99,100,1000\n" +
		"2020-01-06,100,101,99,100,1000\n"

	insp, err := InspectCSV(writeInspectCSV(t, body), "TEST")
	if err != nil {
		t.Fatalf("InspectCSV error = %v", err)
	}
	if !insp.OK() {
		t.Errorf("OK() = false, want true; problems=%+v quality=%s", insp.RowProblems, insp.Quality.String())
	}
}

// TestInspectCSV_BadHeaderStillSurveysRows: a wrong header is reported but the
// positional row survey still runs, so the tool diagnoses both at once.
func TestInspectCSV_BadHeaderStillSurveysRows(t *testing.T) {
	body := "Date,Open,High,Low,Close,Volume\n" + // wrong case → header problem
		"2020-01-02,100,90,110,100,1000\n" //        high<low structural problem

	insp, err := InspectCSV(writeInspectCSV(t, body), "TEST")
	if err != nil {
		t.Fatalf("InspectCSV error = %v", err)
	}
	if insp.HeaderProblem == "" {
		t.Error("HeaderProblem empty, want a mismatch report")
	}
	if len(insp.RowProblems) != 1 {
		t.Errorf("RowProblems = %+v, want 1 (the high<low row surveyed despite bad header)", insp.RowProblems)
	}
	if insp.OK() {
		t.Error("OK() = true, want false")
	}
}

// TestInspectCSV_ZeroVolumeStaysOK: zero-volume bars alone do not fail OK
// (many free sources report 0 for illiquid days/indices — same rule as
// QualityReport.Clean).
func TestInspectCSV_ZeroVolumeStaysOK(t *testing.T) {
	body := "date,open,high,low,close,volume\n" +
		"2020-01-02,100,101,99,100,0\n" +
		"2020-01-03,100,101,99,100,0\n"

	insp, err := InspectCSV(writeInspectCSV(t, body), "TEST")
	if err != nil {
		t.Fatalf("InspectCSV error = %v", err)
	}
	if insp.Quality.ZeroVolumeBars != 2 {
		t.Errorf("ZeroVolumeBars = %d, want 2", insp.Quality.ZeroVolumeBars)
	}
	if !insp.OK() {
		t.Error("OK() = false, want true (zero volume alone is not a failure)")
	}
}
