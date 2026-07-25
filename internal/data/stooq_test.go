package data

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func newStooqTestServer(t *testing.T, handler http.HandlerFunc) (*StooqClient, *httptest.Server) {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)

	client := &StooqClient{
		HTTPClient: srv.Client(),
		BaseURL:    srv.URL + "/",
	}
	return client, srv
}

func TestStooqClient_Bars_HappyPath(t *testing.T) {
	const body = `Date,Open,High,Low,Close,Volume
2020-01-02,100.00,102.00,99.00,101.00,1000
2020-01-03,101.00,103.00,100.50,102.50,1200
`
	client, _ := newStooqTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(body))
	})

	bars, err := client.Bars(context.Background(), "spy")
	if err != nil {
		t.Fatalf("Bars() error = %v", err)
	}
	if len(bars) != 2 {
		t.Fatalf("len(bars) = %d, want 2", len(bars))
	}
	if bars[0].Symbol != "SPY" {
		t.Errorf("bars[0].Symbol = %q, want SPY", bars[0].Symbol)
	}
	if bars[len(bars)-1].Symbol != "SPY" {
		t.Errorf("last bar Symbol = %q, want SPY", bars[len(bars)-1].Symbol)
	}
	if bars[0].Close != 101.00 {
		t.Errorf("bars[0].Close = %v, want 101.00", bars[0].Close)
	}
	if bars[1].Close != 102.50 {
		t.Errorf("bars[1].Close = %v, want 102.50", bars[1].Close)
	}
	if bars[0].Volume != 1000 {
		t.Errorf("bars[0].Volume = %d, want 1000", bars[0].Volume)
	}
	wantTime := time.Date(2020, 1, 2, 0, 0, 0, 0, time.UTC)
	if !bars[0].Time.Equal(wantTime) {
		t.Errorf("bars[0].Time = %v, want %v", bars[0].Time, wantTime)
	}
	if bars[0].Time.Location() != time.UTC {
		t.Errorf("bars[0].Time location = %v, want UTC", bars[0].Time.Location())
	}
}

func TestStooqClient_SymbolMapping(t *testing.T) {
	tests := []struct {
		name  string
		input string
		wantS string
	}{
		{name: "plain ticker gets .us and lowercase", input: "SPY", wantS: "spy.us"},
		{name: "mixed case ticker", input: "AaPl", wantS: "aapl.us"},
		{name: "dotted symbol passes through lowercased", input: "SPY.US", wantS: "spy.us"},
		{name: "caret index passes through lowercased", input: "^SPX", wantS: "^spx"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var gotS string
			client, _ := newStooqTestServer(t, func(w http.ResponseWriter, r *http.Request) {
				gotS = r.URL.Query().Get("s")
				w.Write([]byte("Date,Open,High,Low,Close,Volume\n2020-01-02,1,2,0.5,1.5,100\n"))
			})

			if _, err := client.Bars(context.Background(), tc.input); err != nil {
				t.Fatalf("Bars() error = %v", err)
			}
			if gotS != tc.wantS {
				t.Errorf("query param s = %q, want %q", gotS, tc.wantS)
			}
		})
	}
}

func TestStooqClient_NoVolumeHeader(t *testing.T) {
	const body = `Date,Open,High,Low,Close
2020-01-02,100.00,102.00,99.00,101.00
2020-01-03,101.00,103.00,100.50,102.50
`
	client, _ := newStooqTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(body))
	})

	bars, err := client.Bars(context.Background(), "^spx")
	if err != nil {
		t.Fatalf("Bars() error = %v", err)
	}
	if len(bars) != 2 {
		t.Fatalf("len(bars) = %d, want 2", len(bars))
	}
	for i, b := range bars {
		if b.Volume != 0 {
			t.Errorf("bars[%d].Volume = %d, want 0", i, b.Volume)
		}
	}
}

func TestStooqClient_VolumeEmptyOrDash(t *testing.T) {
	tests := []struct {
		name string
		vol  string
	}{
		{name: "empty volume field", vol: ""},
		{name: "dash volume field", vol: "-"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			body := "Date,Open,High,Low,Close,Volume\n2020-01-02,100.00,102.00,99.00,101.00," + tc.vol + "\n"
			client, _ := newStooqTestServer(t, func(w http.ResponseWriter, r *http.Request) {
				w.Write([]byte(body))
			})

			bars, err := client.Bars(context.Background(), "spy")
			if err != nil {
				t.Fatalf("Bars() error = %v", err)
			}
			if len(bars) != 1 {
				t.Fatalf("len(bars) = %d, want 1", len(bars))
			}
			if bars[0].Volume != 0 {
				t.Errorf("Volume = %d, want 0", bars[0].Volume)
			}
		})
	}
}

func TestStooqClient_NoDataBody(t *testing.T) {
	client, _ := newStooqTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("No data"))
	})

	_, err := client.Bars(context.Background(), "bogus")
	if err == nil {
		t.Fatal("Bars() error = nil, want an error")
	}
	if !strings.Contains(err.Error(), "bogus") {
		t.Errorf("error = %q, want it to mention the symbol %q", err.Error(), "bogus")
	}
}

func TestStooqClient_HTMLBody(t *testing.T) {
	client, _ := newStooqTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("<html><body>rate limited</body></html>"))
	})

	_, err := client.Bars(context.Background(), "spy")
	if err == nil {
		t.Fatal("Bars() error = nil, want an error")
	}
	lower := strings.ToLower(err.Error())
	if !strings.Contains(lower, "limit") && !strings.Contains(lower, "html") {
		t.Errorf("error = %q, want it to mention the rate/download limit or HTML", err.Error())
	}
}

func TestStooqClient_WrongHeader(t *testing.T) {
	client, _ := newStooqTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("Foo,Bar\n1,2\n"))
	})

	_, err := client.Bars(context.Background(), "spy")
	if err == nil {
		t.Fatal("Bars() error = nil, want an error")
	}
}

func TestStooqClient_NonOKStatus(t *testing.T) {
	client, _ := newStooqTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("boom"))
	})

	_, err := client.Bars(context.Background(), "spy")
	if err == nil {
		t.Fatal("Bars() error = nil, want an error")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("error = %q, want it to mention status 500", err.Error())
	}
}

func TestStooqClient_OutOfOrderRows(t *testing.T) {
	const body = `Date,Open,High,Low,Close,Volume
2020-01-03,100.00,102.00,99.00,101.00,1000
2020-01-02,101.00,103.00,100.50,102.50,1200
`
	client, _ := newStooqTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(body))
	})

	_, err := client.Bars(context.Background(), "spy")
	if err == nil {
		t.Fatal("Bars() error = nil, want an error")
	}
}

func TestStooqClient_InvalidOHLCRow(t *testing.T) {
	const body = `Date,Open,High,Low,Close,Volume
2020-01-02,100.00,95.00,99.00,97.00,1000
`
	client, _ := newStooqTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(body))
	})

	_, err := client.Bars(context.Background(), "spy")
	if err == nil {
		t.Fatal("Bars() error = nil, want an error (high < low)")
	}
}

func writeTempStooqCSV(t *testing.T, contents string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "stooq.csv")
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatalf("failed to write temp CSV: %v", err)
	}
	return path
}

func TestLoadStooqCSV_HappyPath(t *testing.T) {
	const body = `Date,Open,High,Low,Close,Volume
2020-01-02,100.00,102.00,99.00,101.00,1000
2020-01-03,101.00,103.00,100.50,102.50,1200
`
	path := writeTempStooqCSV(t, body)

	bars, err := LoadStooqCSV(path, "spy")
	if err != nil {
		t.Fatalf("LoadStooqCSV() error = %v", err)
	}
	if len(bars) != 2 {
		t.Fatalf("len(bars) = %d, want 2", len(bars))
	}
	if bars[0].Symbol != "SPY" {
		t.Errorf("Symbol = %q, want SPY", bars[0].Symbol)
	}
	if bars[0].Close != 101.00 {
		t.Errorf("bars[0].Close = %v, want 101.00", bars[0].Close)
	}
	wantTime := time.Date(2020, 1, 2, 0, 0, 0, 0, time.UTC)
	if !bars[0].Time.Equal(wantTime) {
		t.Errorf("bars[0].Time = %v, want %v", bars[0].Time, wantTime)
	}
}

func TestLoadStooqCSV_MissingFile(t *testing.T) {
	_, err := LoadStooqCSV(filepath.Join(t.TempDir(), "does-not-exist.csv"), "SPY")
	if err == nil {
		t.Fatal("LoadStooqCSV() error = nil, want an error for missing file")
	}
}

func TestLoadStooqCSV_MalformedHeader(t *testing.T) {
	path := writeTempStooqCSV(t, "Foo,Bar\n1,2\n")

	_, err := LoadStooqCSV(path, "SPY")
	if err == nil {
		t.Fatal("LoadStooqCSV() error = nil, want an error for malformed header")
	}
}

func TestNewStooqClient_Defaults(t *testing.T) {
	c := NewStooqClient()
	if c.HTTPClient == nil {
		t.Fatal("HTTPClient is nil")
	}
	if c.HTTPClient.Timeout != 30*time.Second {
		t.Errorf("HTTPClient.Timeout = %v, want 30s", c.HTTPClient.Timeout)
	}
	if c.BaseURL != "https://stooq.com/q/d/l/" {
		t.Errorf("BaseURL = %q, want the Stooq daily endpoint", c.BaseURL)
	}
}
