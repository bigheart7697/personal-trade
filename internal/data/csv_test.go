package data

import (
	"os"
	"path/filepath"
	"testing"
)

func writeTempCSV(t *testing.T, contents string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "bars.csv")
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatalf("failed to write temp CSV: %v", err)
	}
	return path
}

func TestLoadCSV_Valid(t *testing.T) {
	contents := `date,open,high,low,close,volume
2020-01-02,100.00,102.00,99.00,101.00,1000
2020-01-03,101.00,103.00,100.50,102.50,1200
`
	path := writeTempCSV(t, contents)

	bars, err := LoadCSV(path, "TEST")
	if err != nil {
		t.Fatalf("LoadCSV() error = %v", err)
	}
	if len(bars) != 2 {
		t.Fatalf("len(bars) = %d, want 2", len(bars))
	}
	if bars[0].Symbol != "TEST" {
		t.Errorf("Symbol = %q, want TEST", bars[0].Symbol)
	}
	if bars[0].Close != 101.00 {
		t.Errorf("bars[0].Close = %v, want 101.00", bars[0].Close)
	}
}

func TestLoadCSV_Errors(t *testing.T) {
	tests := []struct {
		name     string
		contents string
	}{
		{
			name:     "bad header",
			contents: "d,o,h,l,c,v\n2020-01-02,100,102,99,101,1000\n",
		},
		{
			name: "out of chronological order",
			contents: `date,open,high,low,close,volume
2020-01-03,100.00,102.00,99.00,101.00,1000
2020-01-02,101.00,103.00,100.50,102.50,1200
`,
		},
		{
			name: "non-positive price",
			contents: `date,open,high,low,close,volume
2020-01-02,0.00,102.00,99.00,101.00,1000
`,
		},
		{
			name: "high below low",
			contents: `date,open,high,low,close,volume
2020-01-02,100.00,95.00,99.00,97.00,1000
`,
		},
		{
			name: "high below open",
			contents: `date,open,high,low,close,volume
2020-01-02,110.00,105.00,95.00,100.00,1000
`,
		},
		{
			name: "invalid date",
			contents: `date,open,high,low,close,volume
not-a-date,100.00,102.00,99.00,101.00,1000
`,
		},
		{
			name:     "no data rows",
			contents: "date,open,high,low,close,volume\n",
		},
		{
			name: "duplicate date not strictly increasing",
			contents: `date,open,high,low,close,volume
2020-01-02,100.00,102.00,99.00,101.00,1000
2020-01-02,101.00,103.00,100.50,102.50,1200
`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			path := writeTempCSV(t, tc.contents)
			_, err := LoadCSV(path, "TEST")
			if err == nil {
				t.Fatalf("LoadCSV() error = nil, want an error")
			}
		})
	}
}

func TestLoadCSV_MissingFile(t *testing.T) {
	_, err := LoadCSV(filepath.Join(t.TempDir(), "does-not-exist.csv"), "TEST")
	if err == nil {
		t.Fatal("LoadCSV() error = nil, want an error for missing file")
	}
}
