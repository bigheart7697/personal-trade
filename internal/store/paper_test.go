package store

import (
	"testing"
	"time"
)

func TestSaveAndListPaperRuns_RoundTrip(t *testing.T) {
	st := openTestStore(t)

	ts := time.Date(2026, 7, 6, 14, 30, 0, 0, time.UTC)
	id, err := st.SavePaperRun(ts, "dry-run", "U1234567", 100_000_000_000, 50_000_000_000, `{"note":"test"}`)
	if err != nil {
		t.Fatalf("SavePaperRun() error = %v", err)
	}
	if id == 0 {
		t.Fatal("SavePaperRun() id = 0, want non-zero")
	}

	runs, err := st.ListPaperRuns(10)
	if err != nil {
		t.Fatalf("ListPaperRuns() error = %v", err)
	}
	if len(runs) != 1 {
		t.Fatalf("len(runs) = %d, want 1", len(runs))
	}
	r := runs[0]
	if r.Id != id || r.Mode != "dry-run" || r.Account != "U1234567" {
		t.Errorf("runs[0] = %+v, want Id=%d Mode=dry-run Account=U1234567", r, id)
	}
	if r.EquityMicros != 100_000_000_000 || r.CashMicros != 50_000_000_000 {
		t.Errorf("runs[0] micros = equity=%d cash=%d, want 100000000000/50000000000", r.EquityMicros, r.CashMicros)
	}
	if !r.Ts.Equal(ts) {
		t.Errorf("runs[0].Ts = %v, want %v", r.Ts, ts)
	}
	if r.Detail != `{"note":"test"}` {
		t.Errorf("runs[0].Detail = %q, want the round-tripped JSON", r.Detail)
	}
}

func TestListPaperRuns_NewestFirst(t *testing.T) {
	st := openTestStore(t)

	base := time.Date(2026, 7, 1, 0, 0, 0, 0, time.UTC)
	var lastID int64
	for i := 0; i < 3; i++ {
		id, err := st.SavePaperRun(base.AddDate(0, 0, i), "dry-run", "U1", 0, 0, "{}")
		if err != nil {
			t.Fatalf("SavePaperRun() error = %v", err)
		}
		lastID = id
	}

	runs, err := st.ListPaperRuns(10)
	if err != nil {
		t.Fatalf("ListPaperRuns() error = %v", err)
	}
	if len(runs) != 3 {
		t.Fatalf("len(runs) = %d, want 3", len(runs))
	}
	if runs[0].Id != lastID {
		t.Errorf("runs[0].Id = %d, want %d (newest first)", runs[0].Id, lastID)
	}
}

func TestListPaperRuns_LimitZeroMeansNoLimit(t *testing.T) {
	st := openTestStore(t)
	for i := 0; i < 5; i++ {
		if _, err := st.SavePaperRun(time.Now().UTC(), "dry-run", "U1", 0, 0, "{}"); err != nil {
			t.Fatalf("SavePaperRun() error = %v", err)
		}
	}
	runs, err := st.ListPaperRuns(0)
	if err != nil {
		t.Fatalf("ListPaperRuns() error = %v", err)
	}
	if len(runs) != 5 {
		t.Errorf("len(runs) = %d, want 5", len(runs))
	}
}

func TestSaveAndListPaperOrders_RoundTrip(t *testing.T) {
	st := openTestStore(t)

	runID, err := st.SavePaperRun(time.Now().UTC(), "execute", "U1234567", 100_000_000_000, 50_000_000_000, "{}")
	if err != nil {
		t.Fatalf("SavePaperRun() error = %v", err)
	}

	ts := time.Date(2026, 7, 6, 14, 31, 0, 0, time.UTC)
	orderID, err := st.SavePaperOrder(runID, ts, "dual-momentum", "SPY", "BUY", 10, 450_000_000, "filled", "1000", "detail text")
	if err != nil {
		t.Fatalf("SavePaperOrder() error = %v", err)
	}
	if orderID == 0 {
		t.Fatal("SavePaperOrder() id = 0, want non-zero")
	}

	orders, err := st.ListPaperOrders(runID)
	if err != nil {
		t.Fatalf("ListPaperOrders() error = %v", err)
	}
	if len(orders) != 1 {
		t.Fatalf("len(orders) = %d, want 1", len(orders))
	}
	o := orders[0]
	if o.RunID != runID || o.Strategy != "dual-momentum" || o.Symbol != "SPY" || o.Side != "BUY" {
		t.Errorf("orders[0] = %+v, unexpected fields", o)
	}
	if o.Qty != 10 || o.RefPriceMicros != 450_000_000 {
		t.Errorf("orders[0] qty/price = %d/%d, want 10/450000000", o.Qty, o.RefPriceMicros)
	}
	if o.Status != "filled" || o.OrderID != "1000" || o.Detail != "detail text" {
		t.Errorf("orders[0] status/orderID/detail = %q/%q/%q", o.Status, o.OrderID, o.Detail)
	}
}

func TestListPaperOrders_ScopedToRun(t *testing.T) {
	st := openTestStore(t)

	run1, _ := st.SavePaperRun(time.Now().UTC(), "dry-run", "U1", 0, 0, "{}")
	run2, _ := st.SavePaperRun(time.Now().UTC(), "dry-run", "U1", 0, 0, "{}")

	if _, err := st.SavePaperOrder(run1, time.Now().UTC(), "s1", "AAPL", "BUY", 1, 0, "dry-run", "", ""); err != nil {
		t.Fatalf("SavePaperOrder() error = %v", err)
	}
	if _, err := st.SavePaperOrder(run2, time.Now().UTC(), "s2", "MSFT", "SELL", 2, 0, "dry-run", "", ""); err != nil {
		t.Fatalf("SavePaperOrder() error = %v", err)
	}

	orders1, err := st.ListPaperOrders(run1)
	if err != nil {
		t.Fatalf("ListPaperOrders(run1) error = %v", err)
	}
	if len(orders1) != 1 || orders1[0].Symbol != "AAPL" {
		t.Errorf("ListPaperOrders(run1) = %+v, want exactly 1 AAPL order", orders1)
	}

	orders2, err := st.ListPaperOrders(run2)
	if err != nil {
		t.Fatalf("ListPaperOrders(run2) error = %v", err)
	}
	if len(orders2) != 1 || orders2[0].Symbol != "MSFT" {
		t.Errorf("ListPaperOrders(run2) = %+v, want exactly 1 MSFT order", orders2)
	}
}

func TestListPaperRuns_EmptyDB(t *testing.T) {
	st := openTestStore(t)
	runs, err := st.ListPaperRuns(10)
	if err != nil {
		t.Fatalf("ListPaperRuns() error = %v", err)
	}
	if len(runs) != 0 {
		t.Errorf("len(runs) = %d, want 0", len(runs))
	}
}

func TestLatestPaperEquityPeak_EmptyDB(t *testing.T) {
	st := openTestStore(t)
	peak, ok, err := st.LatestPaperEquityPeak()
	if err != nil {
		t.Fatalf("LatestPaperEquityPeak() error = %v", err)
	}
	if ok {
		t.Errorf("LatestPaperEquityPeak() ok = true, want false for empty DB")
	}
	if peak != 0 {
		t.Errorf("LatestPaperEquityPeak() peak = %d, want 0", peak)
	}
}

func TestLatestPaperEquityPeak_ReturnsMax(t *testing.T) {
	st := openTestStore(t)

	equities := []int64{100_000_000_000, 150_000_000_000, 120_000_000_000}
	for _, eq := range equities {
		if _, err := st.SavePaperRun(time.Now().UTC(), "dry-run", "U1", eq, 0, "{}"); err != nil {
			t.Fatalf("SavePaperRun() error = %v", err)
		}
	}

	peak, ok, err := st.LatestPaperEquityPeak()
	if err != nil {
		t.Fatalf("LatestPaperEquityPeak() error = %v", err)
	}
	if !ok {
		t.Fatal("LatestPaperEquityPeak() ok = false, want true")
	}
	if peak != 150_000_000_000 {
		t.Errorf("LatestPaperEquityPeak() = %d, want 150000000000 (the max)", peak)
	}
}

func TestSetPaperRunAgent_RoundTrip(t *testing.T) {
	st := openTestStore(t)

	id, err := st.SavePaperRun(time.Now().UTC(), "dry-run", "U1", 0, 0, "{}")
	if err != nil {
		t.Fatalf("SavePaperRun() error = %v", err)
	}

	// Freshly saved run: both agent columns default to "".
	runs, err := st.ListPaperRuns(1)
	if err != nil {
		t.Fatalf("ListPaperRuns() error = %v", err)
	}
	if runs[0].TraceJSON != "" || runs[0].AdvisoryJSON != "" {
		t.Errorf("fresh run agent columns = (%q, %q), want both empty", runs[0].TraceJSON, runs[0].AdvisoryJSON)
	}

	trace := `{"regime":"bull","regimeScale":1}`
	advisory := `{"enabled":true,"summary":"ok"}`
	if err := st.SetPaperRunAgent(id, trace, advisory); err != nil {
		t.Fatalf("SetPaperRunAgent() error = %v", err)
	}

	runs, err = st.ListPaperRuns(1)
	if err != nil {
		t.Fatalf("ListPaperRuns() error = %v", err)
	}
	if runs[0].TraceJSON != trace {
		t.Errorf("TraceJSON = %q, want %q", runs[0].TraceJSON, trace)
	}
	if runs[0].AdvisoryJSON != advisory {
		t.Errorf("AdvisoryJSON = %q, want %q", runs[0].AdvisoryJSON, advisory)
	}
}

func TestGetPaperRun_ByID(t *testing.T) {
	st := openTestStore(t)

	id1, err := st.SavePaperRun(time.Now().UTC(), "dry-run", "U1", 100, 50, "{}")
	if err != nil {
		t.Fatalf("SavePaperRun() error = %v", err)
	}
	if err := st.SetPaperRunAgent(id1, `{"regime":"bull"}`, ""); err != nil {
		t.Fatalf("SetPaperRunAgent() error = %v", err)
	}
	// A newer run exists — GetPaperRun(id1) must still return id1's row, not
	// the newest (the "concurrent session interleaves" case).
	id2, err := st.SavePaperRun(time.Now().UTC(), "execute", "U1", 200, 75, "{}")
	if err != nil {
		t.Fatalf("SavePaperRun() error = %v", err)
	}

	run, ok, err := st.GetPaperRun(id1)
	if err != nil || !ok {
		t.Fatalf("GetPaperRun(%d) = ok=%v, err=%v; want found", id1, ok, err)
	}
	if run.Id != id1 || run.Mode != "dry-run" || run.TraceJSON != `{"regime":"bull"}` {
		t.Errorf("GetPaperRun(%d) = id %d mode %q trace %q; want id %d mode dry-run with trace", id1, run.Id, run.Mode, run.TraceJSON, id1)
	}

	if _, ok, err := st.GetPaperRun(id2 + 999); err != nil || ok {
		t.Errorf("GetPaperRun(missing) = ok=%v, err=%v; want (false, nil)", ok, err)
	}
}
