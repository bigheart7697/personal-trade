package paper

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"tradeforge/internal/agent"
	"tradeforge/internal/broker/ibkr"
	"tradeforge/internal/config"
	"tradeforge/internal/domain"
	"tradeforge/internal/store"
	"tradeforge/internal/strategy"
)

// --- test fixtures ---

// fixedSignalStrategy is a minimal single-symbol test strategy: it always
// emits the configured signal on every call to OnBar. It never consults the
// registry — tests inject it via Session.GetStrategy.
type fixedSignalStrategy struct {
	name    string
	signals []domain.Signal
}

func (f *fixedSignalStrategy) Name() string              { return f.name }
func (f *fixedSignalStrategy) Description() string       { return "test fixture" }
func (f *fixedSignalStrategy) Horizon() strategy.Horizon { return strategy.Short }
func (f *fixedSignalStrategy) WarmupBars() int           { return 0 }
func (f *fixedSignalStrategy) OnBar(ctx *strategy.Context, bar domain.Bar) []domain.Signal {
	return f.signals
}

// noSignalStrategy never trades; used for the stale-data-warning test where
// asserting on order behavior isn't the point.
type noSignalStrategy struct{ name string }

func (n *noSignalStrategy) Name() string              { return n.name }
func (n *noSignalStrategy) Description() string       { return "test fixture" }
func (n *noSignalStrategy) Horizon() strategy.Horizon { return strategy.Short }
func (n *noSignalStrategy) WarmupBars() int           { return 0 }
func (n *noSignalStrategy) OnBar(ctx *strategy.Context, bar domain.Bar) []domain.Signal {
	return nil
}

func writeTestCSV(t *testing.T, dir, symbol string, bars []domain.Bar) {
	t.Helper()
	path := filepath.Join(dir, symbol+".csv")
	var sb strings.Builder
	sb.WriteString("date,open,high,low,close,volume\n")
	for _, b := range bars {
		sb.WriteString(b.Time.Format("2006-01-02"))
		sb.WriteString(",")
		sb.WriteString(floatStr(b.Open))
		sb.WriteString(",")
		sb.WriteString(floatStr(b.High))
		sb.WriteString(",")
		sb.WriteString(floatStr(b.Low))
		sb.WriteString(",")
		sb.WriteString(floatStr(b.Close))
		sb.WriteString(",1000\n")
	}
	if err := os.WriteFile(path, []byte(sb.String()), 0o644); err != nil {
		t.Fatalf("writing test CSV: %v", err)
	}
}

func floatStr(f float64) string {
	return strconv.FormatFloat(f, 'f', 4, 64)
}

func freshBars(sym string, n int, start time.Time, price float64) []domain.Bar {
	bars := make([]domain.Bar, n)
	for i := 0; i < n; i++ {
		t := start.AddDate(0, 0, i)
		bars[i] = domain.Bar{
			Symbol: sym,
			Time:   t,
			Open:   price,
			High:   price * 1.01,
			Low:    price * 0.99,
			Close:  price,
			Volume: 1000,
		}
	}
	return bars
}

// fakeGateway is a minimal httptest-backed IBKR Client Portal Gateway
// double, tracking whether /orders and /iserver/exchangerate were ever
// called (for dry-run and FX-call assertions) and letting each test script
// canned responses per path.
type fakeGateway struct {
	srv *httptest.Server

	ordersCalls       int32
	exchangeRateCalls int32

	authenticated bool
	accounts      string // JSON
	positions     string // JSON
	cashLedger    string // JSON

	// exchangeRateResponse is returned verbatim by GET
	// /iserver/exchangerate; if empty, the fake returns 500 so a test that
	// unexpectedly needs FX fails loudly instead of sizing with a silent
	// default rate.
	exchangeRateResponse string

	// placeOrderResponse is returned verbatim by POST .../orders; if empty,
	// a default "no questions" success response is used.
	placeOrderResponse string
	replyResponse      string
	liveOrdersResponse string
}

func newFakeGateway(t *testing.T) *fakeGateway {
	t.Helper()
	fg := &fakeGateway{
		authenticated: true,
		accounts:      `[{"id": "U1234567", "accountVan": "U1234567", "displayName": "Paper", "currency": "USD"}]`,
		positions:     `[]`,
		cashLedger:    `{"USD": {"currency": "USD", "cashbalance": 100000.00}}`,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/iserver/auth/status", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if !fg.authenticated {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		json.NewEncoder(w).Encode(ibkr.AuthStatus{Authenticated: true, Connected: true})
	})
	mux.HandleFunc("/tickle", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"session":"abc"}`))
	})
	mux.HandleFunc("/portfolio/accounts", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(fg.accounts))
	})
	mux.HandleFunc("/portfolio/U1234567/positions/0", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(fg.positions))
	})
	mux.HandleFunc("/portfolio/U1234567/ledger", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(fg.cashLedger))
	})
	mux.HandleFunc("/iserver/exchangerate", func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&fg.exchangeRateCalls, 1)
		if fg.exchangeRateResponse == "" {
			http.Error(w, `{"error":"no exchange rate configured in fakeGateway"}`, http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(fg.exchangeRateResponse))
	})
	mux.HandleFunc("/iserver/secdef/search", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`[{"conid": "265598", "symbol": "SPY", "sections": [{"secType": "STK"}]}]`))
	})
	mux.HandleFunc("/iserver/account/U1234567/orders", func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&fg.ordersCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		if fg.placeOrderResponse != "" {
			w.Write([]byte(fg.placeOrderResponse))
			return
		}
		w.Write([]byte(`[{"order_id": "1000", "order_status": "Submitted"}]`))
	})
	mux.HandleFunc("/iserver/reply/q1", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if fg.replyResponse != "" {
			w.Write([]byte(fg.replyResponse))
			return
		}
		w.Write([]byte(`[{"order_id": "1000", "order_status": "Submitted"}]`))
	})
	mux.HandleFunc("/iserver/account/orders", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if fg.liveOrdersResponse != "" {
			w.Write([]byte(fg.liveOrdersResponse))
			return
		}
		w.Write([]byte(`{"orders": [{"orderId": "1000", "cOID": "x", "conid": 265598, "side": "BUY", "totalSize": 10, "filledQuantity": 10, "avgPrice": 450.0, "status": "Filled"}]}`))
	})

	fg.srv = httptest.NewServer(mux)
	t.Cleanup(fg.srv.Close)
	return fg
}

func (fg *fakeGateway) client() *ibkr.Client {
	return &ibkr.Client{BaseURL: fg.srv.URL, HTTPClient: fg.srv.Client()}
}

func newTestSession(t *testing.T, fg *fakeGateway, cfg config.Config, getStrategy func(string) (strategy.Strategy, error)) (*Session, *strings.Builder) {
	t.Helper()
	dbPath := filepath.Join(t.TempDir(), "test.db")
	st, err := store.Open(dbPath)
	if err != nil {
		t.Fatalf("store.Open() error = %v", err)
	}
	t.Cleanup(func() { st.Close() })

	var out strings.Builder
	sess := &Session{
		Cfg:         cfg,
		Client:      fg.client(),
		Store:       st,
		Out:         &out,
		GetStrategy: getStrategy,
		NowFunc:     func() time.Time { return time.Date(2026, 7, 6, 21, 0, 0, 0, time.UTC) },
	}
	return sess, &out
}

func baseConfig(dataDir string) config.Config {
	return config.Config{
		Mode: "paper",
		Paper: config.PaperConfig{
			Strategies: []string{"test-strat"},
			DataDir:    dataDir,
			AccountID:  "U1234567",
			Symbols:    map[string]string{"test-strat": "SPY"},
		},
	}
}

// --- tests ---

func TestRunOnce_ModeSimRefuses(t *testing.T) {
	fg := newFakeGateway(t)
	cfg := baseConfig(t.TempDir())
	cfg.Mode = "sim"
	sess, _ := newTestSession(t, fg, cfg, nil)

	err := sess.RunOnce(context.Background(), false)
	if err == nil {
		t.Fatal("RunOnce() error = nil, want refusal for sim mode")
	}
	if !strings.Contains(err.Error(), "paper") {
		t.Errorf("RunOnce() error = %q, want it to mention setting mode to paper", err.Error())
	}
}

func TestRunOnce_ModeLiveRefuses(t *testing.T) {
	fg := newFakeGateway(t)
	cfg := baseConfig(t.TempDir())
	cfg.Mode = "live"
	cfg.LiveApproval = &config.LiveApproval{Date: "2099-01-01", Statement: "test-fixture-approval"}
	t.Setenv("TF_CONFIRM_LIVE", "yes")
	sess, _ := newTestSession(t, fg, cfg, nil)

	err := sess.RunOnce(context.Background(), false)
	if err == nil {
		t.Fatal("RunOnce() error = nil, want refusal for live mode")
	}
	if !strings.Contains(err.Error(), "live order path does not exist") {
		t.Errorf("RunOnce() error = %q, want it to mention the live order path doesn't exist", err.Error())
	}
}

func TestRunOnce_Unauthenticated(t *testing.T) {
	fg := newFakeGateway(t)
	fg.authenticated = false
	cfg := baseConfig(t.TempDir())
	sess, _ := newTestSession(t, fg, cfg, nil)

	err := sess.RunOnce(context.Background(), false)
	if err == nil {
		t.Fatal("RunOnce() error = nil, want ErrNotAuthenticated")
	}
	if !strings.Contains(err.Error(), "authenticated") {
		t.Errorf("RunOnce() error = %q, want it to mention authentication", err.Error())
	}
}

func TestRunOnce_DryRun_PlacesNothingButRecordsLedger(t *testing.T) {
	fg := newFakeGateway(t)
	dataDir := t.TempDir()
	writeTestCSV(t, dataDir, "SPY", freshBars("SPY", 30, time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC), 450))

	cfg := baseConfig(dataDir)
	getStrategy := func(name string) (strategy.Strategy, error) {
		return &fixedSignalStrategy{name: "test-strat", signals: []domain.Signal{{Symbol: "SPY", TargetWeight: 0.1}}}, nil
	}
	sess, out := newTestSession(t, fg, cfg, getStrategy)

	if err := sess.RunOnce(context.Background(), false); err != nil {
		t.Fatalf("RunOnce() error = %v", err)
	}

	if atomic.LoadInt32(&fg.ordersCalls) != 0 {
		t.Errorf("gateway /orders called %d times in dry-run, want 0", fg.ordersCalls)
	}
	if !strings.Contains(out.String(), "dry-run") {
		t.Errorf("report output = %q, want it to mention dry-run", out.String())
	}
	if !strings.Contains(out.String(), "PAPER SESSION COMPLETE") {
		t.Errorf("report output missing completion line: %q", out.String())
	}

	runs, err := sess.Store.ListPaperRuns(10)
	if err != nil {
		t.Fatalf("ListPaperRuns() error = %v", err)
	}
	if len(runs) != 1 {
		t.Fatalf("len(runs) = %d, want 1", len(runs))
	}
	if runs[0].Mode != "dry-run" {
		t.Errorf("runs[0].Mode = %q, want dry-run", runs[0].Mode)
	}

	orders, err := sess.Store.ListPaperOrders(runs[0].Id)
	if err != nil {
		t.Fatalf("ListPaperOrders() error = %v", err)
	}
	if len(orders) != 1 {
		t.Fatalf("len(orders) = %d, want 1", len(orders))
	}
	if orders[0].Status != "dry-run" {
		t.Errorf("orders[0].Status = %q, want dry-run", orders[0].Status)
	}
}

func TestRunOnce_Execute_PlacesPollsAndRecords(t *testing.T) {
	fg := newFakeGateway(t)
	dataDir := t.TempDir()
	writeTestCSV(t, dataDir, "SPY", freshBars("SPY", 30, time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC), 450))

	cfg := baseConfig(dataDir)
	getStrategy := func(name string) (strategy.Strategy, error) {
		return &fixedSignalStrategy{name: "test-strat", signals: []domain.Signal{{Symbol: "SPY", TargetWeight: 0.1}}}, nil
	}
	sess, out := newTestSession(t, fg, cfg, getStrategy)

	if err := sess.RunOnce(context.Background(), true); err != nil {
		t.Fatalf("RunOnce() error = %v", err)
	}

	if atomic.LoadInt32(&fg.ordersCalls) != 1 {
		t.Errorf("gateway /orders called %d times, want 1", fg.ordersCalls)
	}
	if !strings.Contains(out.String(), "EXECUTING AGAINST PAPER ACCOUNT") {
		t.Errorf("report output = %q, want the execute banner", out.String())
	}

	runs, err := sess.Store.ListPaperRuns(10)
	if err != nil {
		t.Fatalf("ListPaperRuns() error = %v", err)
	}
	if len(runs) != 1 || runs[0].Mode != "execute" {
		t.Fatalf("runs = %+v, want exactly 1 execute run", runs)
	}

	orders, err := sess.Store.ListPaperOrders(runs[0].Id)
	if err != nil {
		t.Fatalf("ListPaperOrders() error = %v", err)
	}
	if len(orders) != 1 {
		t.Fatalf("len(orders) = %d, want 1", len(orders))
	}
	if orders[0].OrderID != "1000" {
		t.Errorf("orders[0].OrderID = %q, want 1000", orders[0].OrderID)
	}
	if orders[0].Status != "Filled" {
		t.Errorf("orders[0].Status = %q, want Filled (polled)", orders[0].Status)
	}
}

func TestRunOnce_RiskRejection_RecordedNotOrdered(t *testing.T) {
	fg := newFakeGateway(t)
	dataDir := t.TempDir()
	writeTestCSV(t, dataDir, "SPY", freshBars("SPY", 30, time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC), 450))

	cfg := baseConfig(dataDir)
	// TargetWeight 0 with no existing position is a no-op ("already at
	// target") -> rejected by risk.Manager without ever reaching the
	// gateway.
	getStrategy := func(name string) (strategy.Strategy, error) {
		return &fixedSignalStrategy{name: "test-strat", signals: []domain.Signal{{Symbol: "SPY", TargetWeight: 0}}}, nil
	}
	sess, _ := newTestSession(t, fg, cfg, getStrategy)

	if err := sess.RunOnce(context.Background(), true); err != nil {
		t.Fatalf("RunOnce() error = %v", err)
	}

	if atomic.LoadInt32(&fg.ordersCalls) != 0 {
		t.Errorf("gateway /orders called %d times, want 0 (signal should have been rejected)", fg.ordersCalls)
	}

	runs, err := sess.Store.ListPaperRuns(10)
	if err != nil || len(runs) != 1 {
		t.Fatalf("ListPaperRuns() = %+v, err = %v", runs, err)
	}
	orders, err := sess.Store.ListPaperOrders(runs[0].Id)
	if err != nil {
		t.Fatalf("ListPaperOrders() error = %v", err)
	}
	if len(orders) != 1 {
		t.Fatalf("len(orders) = %d, want 1 (rejection still recorded)", len(orders))
	}
	if orders[0].Status != "rejected" {
		t.Errorf("orders[0].Status = %q, want rejected", orders[0].Status)
	}
}

func TestRunOnce_UnknownQuestion_Aborts(t *testing.T) {
	fg := newFakeGateway(t)
	fg.placeOrderResponse = `[{"id": "q1", "message": ["This order will exceed your margin requirements."]}]`
	dataDir := t.TempDir()
	writeTestCSV(t, dataDir, "SPY", freshBars("SPY", 30, time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC), 450))

	cfg := baseConfig(dataDir)
	getStrategy := func(name string) (strategy.Strategy, error) {
		return &fixedSignalStrategy{name: "test-strat", signals: []domain.Signal{{Symbol: "SPY", TargetWeight: 0.1}}}, nil
	}
	sess, out := newTestSession(t, fg, cfg, getStrategy)

	// RunOnce itself does not hard-fail on a per-order placement error (it
	// logs and continues so other strategies' orders still get a chance);
	// the abort surfaces as an "error:" status recorded in the ledger and
	// printed in the report.
	if err := sess.RunOnce(context.Background(), true); err != nil {
		t.Fatalf("RunOnce() error = %v, want nil (per-order errors are recorded, not fatal)", err)
	}

	if !strings.Contains(out.String(), "margin requirements") {
		t.Errorf("report output = %q, want it to surface the unrecognized question", out.String())
	}

	runs, _ := sess.Store.ListPaperRuns(10)
	orders, err := sess.Store.ListPaperOrders(runs[0].Id)
	if err != nil {
		t.Fatalf("ListPaperOrders() error = %v", err)
	}
	if len(orders) != 1 {
		t.Fatalf("len(orders) = %d, want 1", len(orders))
	}
	if !strings.Contains(orders[0].Status, "error") {
		t.Errorf("orders[0].Status = %q, want it to record the error", orders[0].Status)
	}
}

func TestRunOnce_StaleDataWarning(t *testing.T) {
	fg := newFakeGateway(t)
	dataDir := t.TempDir()
	// Bars ending 30 days before "now" (2026-07-06) — well past staleAfter.
	writeTestCSV(t, dataDir, "SPY", freshBars("SPY", 10, time.Date(2026, 5, 1, 0, 0, 0, 0, time.UTC), 450))

	cfg := baseConfig(dataDir)
	getStrategy := func(name string) (strategy.Strategy, error) {
		return &noSignalStrategy{name: "test-strat"}, nil
	}
	sess, out := newTestSession(t, fg, cfg, getStrategy)

	if err := sess.RunOnce(context.Background(), false); err != nil {
		t.Fatalf("RunOnce() error = %v", err)
	}

	if !strings.Contains(out.String(), "stale") {
		t.Errorf("report output = %q, want a stale-data warning", out.String())
	}
}

// TestRunOnce_CADBase_SizesInUSD is the regression test for the FX
// mis-sizing found live 2026-07-07: the session fed ApproveOrder BASE(CAD)
// equity while prices were USD, so weight*equity/price over-sized the first
// order by ~1/fx (280 QQQ ≈ 27% of equity vs the 20% cap). With FX-aware
// sizing the order must be computed from USD equity.
//
// Hand-computed expectation:
//
//	netLiq        = 1,000,000 CAD (BASE ledger)
//	usdRate       = 1.25 CAD per USD (gateway /iserver/exchangerate)
//	netLiqUSD     = 1,000,000 / 1.25            = 800,000 USD
//	sizing book   : SPY 100 sh @ 450 (mktValue 45,000 USD)
//	                Cash = 800,000 - 45,000     = 755,000 USD
//	equity in ApproveOrder = 755,000 + 100*450  = 800,000 USD (exactly netLiqUSD)
//	desiredQty    = floor(0.20 * 800,000 / 450) = floor(355.55) = 355
//	orderQty      = desiredQty - currentQty     = 355 - 100     = 255 BUY
//
// (The old BASE-sized bug would have produced floor(0.20*1,000,000/450) -
// 100 = 344, so 255 discriminates.)
func TestRunOnce_CADBase_SizesInUSD(t *testing.T) {
	fg := newFakeGateway(t)
	fg.accounts = `[{"id": "U1234567", "accountVan": "U1234567", "displayName": "Paper", "currency": "CAD"}]`
	fg.cashLedger = `{"BASE": {"currency": "BASE", "cashbalance": 716773.94, "netliquidationvalue": 1000000.00}}`
	fg.exchangeRateResponse = `{"rate": 1.25}`
	fg.positions = `[{"conid": 265598, "contractDesc": "SPY", "position": 100, "mktValue": 45000.0, "currency": "USD"}]`

	dataDir := t.TempDir()
	writeTestCSV(t, dataDir, "SPY", freshBars("SPY", 30, time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC), 450))

	cfg := baseConfig(dataDir)
	getStrategy := func(name string) (strategy.Strategy, error) {
		return &fixedSignalStrategy{name: "test-strat", signals: []domain.Signal{{Symbol: "SPY", TargetWeight: 0.20}}}, nil
	}
	sess, out := newTestSession(t, fg, cfg, getStrategy)

	if err := sess.RunOnce(context.Background(), false); err != nil {
		t.Fatalf("RunOnce() error = %v", err)
	}

	if got := atomic.LoadInt32(&fg.exchangeRateCalls); got != 1 {
		t.Errorf("gateway /iserver/exchangerate called %d times, want 1", got)
	}

	runs, err := sess.Store.ListPaperRuns(10)
	if err != nil || len(runs) != 1 {
		t.Fatalf("ListPaperRuns() = %+v, err = %v, want exactly 1 run", runs, err)
	}
	// The persisted ledger stays BASE: 1,000,000 CAD = 1e12 micros.
	if runs[0].EquityMicros != 1_000_000_000_000 {
		t.Errorf("runs[0].EquityMicros = %d, want 1000000000000 (BASE-currency micros, not USD)", runs[0].EquityMicros)
	}

	orders, err := sess.Store.ListPaperOrders(runs[0].Id)
	if err != nil {
		t.Fatalf("ListPaperOrders() error = %v", err)
	}
	if len(orders) != 1 {
		t.Fatalf("len(orders) = %d, want 1", len(orders))
	}
	if orders[0].Side != "BUY" || orders[0].Qty != 255 {
		t.Errorf("order = %s %d, want BUY 255 (floor(0.20*800000/450)=355 minus current 100)", orders[0].Side, orders[0].Qty)
	}

	// The report header must show both currencies and the rate.
	if !strings.Contains(out.String(), "800000.00 USD") || !strings.Contains(out.String(), "1.2500 CAD") {
		t.Errorf("report output = %q, want the dual-currency equity line with 800000.00 USD @ 1.2500 CAD", out.String())
	}
	// The interim approximate-sizing warning must be gone.
	if strings.Contains(out.String(), "APPROXIMATE") {
		t.Errorf("report output still contains the interim APPROXIMATE sizing warning: %q", out.String())
	}
}

// TestRunOnce_USDBase_NoFXCall pins the USD-base account behavior: no
// /iserver/exchangerate call at all (usdRate is 1.0 by definition), and
// sizing identical to the old single-currency arithmetic —
// floor(0.10 * 100,000 / 450) = floor(22.22) = 22 BUY from a flat book.
func TestRunOnce_USDBase_NoFXCall(t *testing.T) {
	fg := newFakeGateway(t) // defaults: currency USD, USD ledger cash 100,000
	dataDir := t.TempDir()
	writeTestCSV(t, dataDir, "SPY", freshBars("SPY", 30, time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC), 450))

	cfg := baseConfig(dataDir)
	getStrategy := func(name string) (strategy.Strategy, error) {
		return &fixedSignalStrategy{name: "test-strat", signals: []domain.Signal{{Symbol: "SPY", TargetWeight: 0.10}}}, nil
	}
	sess, out := newTestSession(t, fg, cfg, getStrategy)

	if err := sess.RunOnce(context.Background(), false); err != nil {
		t.Fatalf("RunOnce() error = %v", err)
	}

	if got := atomic.LoadInt32(&fg.exchangeRateCalls); got != 0 {
		t.Errorf("gateway /iserver/exchangerate called %d times for a USD-base account, want 0", got)
	}

	runs, err := sess.Store.ListPaperRuns(10)
	if err != nil || len(runs) != 1 {
		t.Fatalf("ListPaperRuns() = %+v, err = %v, want exactly 1 run", runs, err)
	}
	orders, err := sess.Store.ListPaperOrders(runs[0].Id)
	if err != nil {
		t.Fatalf("ListPaperOrders() error = %v", err)
	}
	if len(orders) != 1 || orders[0].Side != "BUY" || orders[0].Qty != 22 {
		t.Fatalf("orders = %+v, want exactly 1 BUY 22 (floor(0.10*100000/450))", orders)
	}

	// USD accounts keep the unchanged single-currency header form.
	if !strings.Contains(out.String(), "Equity:    $100000.00 (cash $100000.00)") {
		t.Errorf("report output = %q, want the single-currency equity line", out.String())
	}
}

// TestRunOnce_FXFailure_RefusesToRun pins the fail-closed rule: on a non-USD
// account, ANY failure to obtain the FX rate must abort the whole session
// before any signal is computed — mis-sizing is worse than not running.
func TestRunOnce_FXFailure_RefusesToRun(t *testing.T) {
	tests := []struct {
		name         string
		exchangeResp string // "" = fake returns HTTP 500
	}{
		{name: "gateway error (500)", exchangeResp: ""},
		{name: "non-positive rate", exchangeResp: `{"rate": 0}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fg := newFakeGateway(t)
			fg.accounts = `[{"id": "U1234567", "accountVan": "U1234567", "displayName": "Paper", "currency": "CAD"}]`
			fg.cashLedger = `{"BASE": {"currency": "BASE", "cashbalance": 1000000.0, "netliquidationvalue": 1000000.0}}`
			fg.exchangeRateResponse = tt.exchangeResp

			dataDir := t.TempDir()
			writeTestCSV(t, dataDir, "SPY", freshBars("SPY", 30, time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC), 450))

			cfg := baseConfig(dataDir)
			getStrategy := func(name string) (strategy.Strategy, error) {
				return &fixedSignalStrategy{name: "test-strat", signals: []domain.Signal{{Symbol: "SPY", TargetWeight: 0.20}}}, nil
			}
			sess, _ := newTestSession(t, fg, cfg, getStrategy)

			err := sess.RunOnce(context.Background(), true)
			if err == nil {
				t.Fatal("RunOnce() error = nil, want a refusal when the FX rate is unavailable")
			}
			if !strings.Contains(err.Error(), "FX rate") || !strings.Contains(err.Error(), "CAD") {
				t.Errorf("RunOnce() error = %q, want it to name the FX rate and the base currency", err.Error())
			}

			if got := atomic.LoadInt32(&fg.ordersCalls); got != 0 {
				t.Errorf("gateway /orders called %d times, want 0 (session must refuse before any order)", got)
			}
			runs, err := sess.Store.ListPaperRuns(10)
			if err != nil {
				t.Fatalf("ListPaperRuns() error = %v", err)
			}
			if len(runs) != 0 {
				t.Errorf("len(runs) = %d, want 0 (refusal happens before the ledger)", len(runs))
			}
		})
	}
}

// TestRunOnce_KillSwitch_RatioPreservedAcrossFX proves the drawdown
// kill-switch behaves identically to a BASE-currency check after the USD
// conversion: the stored BASE peak and today's BASE equity are both divided
// by the same day's rate, so equityUSD/peakUSD == equityBASE/peakBASE.
//
// Stored peak: 1,000,000 CAD. MaxDrawdown 25% → the line is at 750,000 CAD
// (600,000 USD at 1.25). Just past the line (749,000 CAD = 599,200 USD) an
// exposure-increasing order must be rejected; just inside (751,000 CAD =
// 600,800 USD) it must be approved: floor(0.10*600,800/450) = 133 BUY.
func TestRunOnce_KillSwitch_RatioPreservedAcrossFX(t *testing.T) {
	tests := []struct {
		name         string
		netLiqCAD    string
		wantApproved bool
		wantQty      int64
	}{
		{name: "just past the 25% line: rejected", netLiqCAD: "749000.0", wantApproved: false},
		{name: "just inside the 25% line: approved", netLiqCAD: "751000.0", wantApproved: true, wantQty: 133},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fg := newFakeGateway(t)
			fg.accounts = `[{"id": "U1234567", "accountVan": "U1234567", "displayName": "Paper", "currency": "CAD"}]`
			fg.cashLedger = `{"BASE": {"currency": "BASE", "cashbalance": ` + tt.netLiqCAD + `, "netliquidationvalue": ` + tt.netLiqCAD + `}}`
			fg.exchangeRateResponse = `{"rate": 1.25}`

			dataDir := t.TempDir()
			writeTestCSV(t, dataDir, "SPY", freshBars("SPY", 30, time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC), 450))

			cfg := baseConfig(dataDir)
			getStrategy := func(name string) (strategy.Strategy, error) {
				return &fixedSignalStrategy{name: "test-strat", signals: []domain.Signal{{Symbol: "SPY", TargetWeight: 0.10}}}, nil
			}
			sess, _ := newTestSession(t, fg, cfg, getStrategy)

			// Seed the stored peak: 1,000,000 CAD in BASE micros, exactly as a
			// previous session's recordLedger would have persisted it.
			seedTs := time.Date(2026, 7, 1, 21, 0, 0, 0, time.UTC)
			if _, err := sess.Store.SavePaperRun(seedTs, "dry-run", "U1234567", 1_000_000_000_000, 1_000_000_000_000, "{}"); err != nil {
				t.Fatalf("seeding stored equity peak: %v", err)
			}

			if err := sess.RunOnce(context.Background(), false); err != nil {
				t.Fatalf("RunOnce() error = %v", err)
			}

			runs, err := sess.Store.ListPaperRuns(1)
			if err != nil || len(runs) != 1 {
				t.Fatalf("ListPaperRuns() = %+v, err = %v, want the new run first", runs, err)
			}
			orders, err := sess.Store.ListPaperOrders(runs[0].Id)
			if err != nil {
				t.Fatalf("ListPaperOrders() error = %v", err)
			}
			if len(orders) != 1 {
				t.Fatalf("len(orders) = %d, want 1", len(orders))
			}

			if tt.wantApproved {
				if orders[0].Status != "dry-run" || orders[0].Side != "BUY" || orders[0].Qty != tt.wantQty {
					t.Errorf("order = %+v, want approved BUY %d with status dry-run", orders[0], tt.wantQty)
				}
			} else {
				if orders[0].Status != "rejected" {
					t.Errorf("orders[0].Status = %q, want rejected (kill switch)", orders[0].Status)
				}
				if !strings.Contains(orders[0].Detail, "kill switch") {
					t.Errorf("orders[0].Detail = %q, want the kill-switch rejection reason", orders[0].Detail)
				}
			}
		})
	}
}

func TestRunOnce_NoBaseCurrency_RefusesToRun(t *testing.T) {
	// The gateway not reporting a base currency means the sizing units are
	// unknown; the session must fail closed rather than guess.
	fg := newFakeGateway(t)
	fg.accounts = `[{"id": "U1234567", "accountVan": "U1234567", "displayName": "Paper"}]` // no currency field

	dataDir := t.TempDir()
	writeTestCSV(t, dataDir, "SPY", freshBars("SPY", 30, time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC), 450))

	cfg := baseConfig(dataDir)
	getStrategy := func(name string) (strategy.Strategy, error) {
		return &fixedSignalStrategy{name: "test-strat", signals: []domain.Signal{{Symbol: "SPY", TargetWeight: 0.10}}}, nil
	}
	sess, _ := newTestSession(t, fg, cfg, getStrategy)

	err := sess.RunOnce(context.Background(), true)
	if err == nil {
		t.Fatal("RunOnce() error = nil, want a refusal when the base currency is unknown")
	}
	if !strings.Contains(err.Error(), "base currency") {
		t.Errorf("RunOnce() error = %q, want it to mention the missing base currency", err.Error())
	}
	if got := atomic.LoadInt32(&fg.ordersCalls); got != 0 {
		t.Errorf("gateway /orders called %d times, want 0", got)
	}
}

func TestRunOnce_NoStrategiesConfigured(t *testing.T) {
	fg := newFakeGateway(t)
	cfg := config.Config{Mode: "paper", Paper: config.PaperConfig{DataDir: t.TempDir()}}
	sess, out := newTestSession(t, fg, cfg, nil)

	if err := sess.RunOnce(context.Background(), false); err != nil {
		t.Fatalf("RunOnce() error = %v, want nil (nothing configured is not an error)", err)
	}
	if !strings.Contains(out.String(), "nothing to run") {
		t.Errorf("report output = %q, want a no-strategies message", out.String())
	}
	if atomic.LoadInt32(&fg.ordersCalls) != 0 {
		t.Errorf("gateway /orders called %d times, want 0", fg.ordersCalls)
	}
}

// TestPositionAges_DerivedFromLedger covers the paper-side PositionAge
// supply (the restart-proof replacement for the strategies' old barsHeld
// counters): the opening BUY is found by undoing filled orders newest-first
// until the reconstructed position hits flat, and age counts bars from the
// fill DAY inclusive (first session after entry sees age 1).
func TestPositionAges_DerivedFromLedger(t *testing.T) {
	fg := newFakeGateway(t)
	sess, _ := newTestSession(t, fg, baseConfig(t.TempDir()), nil)

	day := func(d int) time.Time { return time.Date(2026, 7, d, 14, 30, 0, 0, time.UTC) }
	bar := func(d int) domain.Bar {
		ts := time.Date(2026, 7, d, 0, 0, 0, 0, time.UTC)
		return domain.Bar{Symbol: "QQQ", Time: ts, Open: 700, High: 710, Low: 690, Close: 705, Volume: 1}
	}

	// Ledger: run1 opens 100 QQQ (Filled) on July 2; a dry-run row the same
	// day must be ignored; run2 sells 40 (Filled) on July 3. Current qty 60.
	run1, err := sess.Store.SavePaperRun(day(2), "execute", "U1", 1, 1, "{}")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := sess.Store.SavePaperOrder(run1, day(2), "s", "QQQ", "BUY", 100, 1, "Filled", "o1", ""); err != nil {
		t.Fatal(err)
	}
	if _, err := sess.Store.SavePaperOrder(run1, day(2), "s", "QQQ", "BUY", 999, 1, "dry-run", "", ""); err != nil {
		t.Fatal(err)
	}
	run2, err := sess.Store.SavePaperRun(day(3), "execute", "U1", 1, 1, "{}")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := sess.Store.SavePaperOrder(run2, day(3), "s", "QQQ", "SELL", 40, 1, "Filled", "o2", ""); err != nil {
		t.Fatal(err)
	}

	port := domain.NewPortfolio(1000)
	port.Positions["QQQ"] = domain.Position{Symbol: "QQQ", Qty: 60, AvgPrice: 700}

	// Bars July 1..5: entry day July 2 counts, so age = 4 (July 2,3,4,5).
	bars := []domain.Bar{bar(1), bar(2), bar(3), bar(4), bar(5)}
	ages := sess.positionAges(port, map[string][]domain.Bar{"QQQ": bars})
	if ages["QQQ"] != 4 {
		t.Errorf("PositionAge(QQQ) = %d, want 4 (bars on/after the July-2 opening fill)", ages["QQQ"])
	}

	// A symbol with no ledger trail resolves to 0 = unknown (safe degradation).
	port.Positions["SPY"] = domain.Position{Symbol: "SPY", Qty: 10, AvgPrice: 500}
	ages = sess.positionAges(port, map[string][]domain.Bar{"QQQ": bars, "SPY": bars})
	if ages["SPY"] != 0 {
		t.Errorf("PositionAge(SPY) = %d, want 0 (no ledger trail)", ages["SPY"])
	}
}

// --- agentic layer: trace capture + advisory ---

// tracedStrategy is a fixedSignalStrategy that also implements
// strategy.Traced, returning a canned trace only after OnBar has run
// (mirroring agent-committee's record-during-OnUniverseBar behavior).
type tracedStrategy struct {
	fixedSignalStrategy
	ran   bool
	trace string
}

func (ts *tracedStrategy) OnBar(ctx *strategy.Context, bar domain.Bar) []domain.Signal {
	ts.ran = true
	return ts.fixedSignalStrategy.OnBar(ctx, bar)
}

func (ts *tracedStrategy) TraceJSON() ([]byte, error) {
	if !ts.ran {
		return nil, nil
	}
	return []byte(ts.trace), nil
}

var _ strategy.Traced = (*tracedStrategy)(nil)

// stubAdvisor is a canned agent.Provider capturing the ReviewInput it was
// handed.
type stubAdvisor struct {
	advisory agent.Advisory
	gotInput *agent.ReviewInput
}

func (sa *stubAdvisor) Name() string { return "stub" }
func (sa *stubAdvisor) Review(ctx context.Context, in agent.ReviewInput) (agent.Advisory, error) {
	sa.gotInput = &in
	return sa.advisory, nil
}

func TestRunOnce_CapturesTraceAndAdvisoryIntoLedger(t *testing.T) {
	fg := newFakeGateway(t)
	dataDir := t.TempDir()
	writeTestCSV(t, dataDir, "SPY", freshBars("SPY", 30, time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC), 450))

	cfg := baseConfig(dataDir)
	wantTrace := `{"regime":"bull","regimeScale":1,"finalWeights":{"SPY":0.1}}`
	getStrategy := func(name string) (strategy.Strategy, error) {
		return &tracedStrategy{
			fixedSignalStrategy: fixedSignalStrategy{name: "test-strat", signals: []domain.Signal{{Symbol: "SPY", TargetWeight: 0.1}}},
			trace:               wantTrace,
		}, nil
	}
	sess, out := newTestSession(t, fg, cfg, getStrategy)
	advisor := &stubAdvisor{advisory: agent.Advisory{
		Enabled:    true,
		Model:      "test-model",
		Summary:    "plan looks reasonable",
		Warnings:   []string{"concentration: single-symbol plan"},
		Confidence: "medium",
		TokensIn:   100,
		TokensOut:  30,
	}}
	sess.Advisor = advisor

	if err := sess.RunOnce(context.Background(), false); err != nil {
		t.Fatalf("RunOnce() error = %v", err)
	}

	// The report gains an AGENT ADVISORY section with the summary + warning.
	report := out.String()
	if !strings.Contains(report, "AGENT ADVISORY") {
		t.Errorf("report missing AGENT ADVISORY section: %q", report)
	}
	if !strings.Contains(report, "plan looks reasonable") {
		t.Errorf("report missing advisory summary: %q", report)
	}
	if !strings.Contains(report, "concentration: single-symbol plan") {
		t.Errorf("report missing advisory warning: %q", report)
	}

	// The advisor saw the plan and the trace.
	if advisor.gotInput == nil {
		t.Fatal("advisor was never consulted")
	}
	if advisor.gotInput.CommitteeTraceJSON != wantTrace {
		t.Errorf("ReviewInput.CommitteeTraceJSON = %q, want %q", advisor.gotInput.CommitteeTraceJSON, wantTrace)
	}
	if len(advisor.gotInput.PlannedOrders) != 1 || advisor.gotInput.PlannedOrders[0].Symbol != "SPY" {
		t.Errorf("ReviewInput.PlannedOrders = %+v, want one SPY order", advisor.gotInput.PlannedOrders)
	}

	// Both artifacts persisted on the run row.
	runs, err := sess.Store.ListPaperRuns(1)
	if err != nil {
		t.Fatalf("ListPaperRuns() error = %v", err)
	}
	if len(runs) != 1 {
		t.Fatalf("len(runs) = %d, want 1", len(runs))
	}
	if runs[0].TraceJSON != wantTrace {
		t.Errorf("persisted TraceJSON = %q, want %q", runs[0].TraceJSON, wantTrace)
	}
	var persisted agent.Advisory
	if err := json.Unmarshal([]byte(runs[0].AdvisoryJSON), &persisted); err != nil {
		t.Fatalf("persisted AdvisoryJSON does not parse: %v (%q)", err, runs[0].AdvisoryJSON)
	}
	if !persisted.Enabled || persisted.Summary != "plan looks reasonable" {
		t.Errorf("persisted advisory = %+v, want the stub's advisory round-tripped", persisted)
	}
}

func TestRunOnce_NullAdvisorReportsDisabledOneLiner(t *testing.T) {
	fg := newFakeGateway(t)
	dataDir := t.TempDir()
	writeTestCSV(t, dataDir, "SPY", freshBars("SPY", 30, time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC), 450))

	cfg := baseConfig(dataDir)
	getStrategy := func(name string) (strategy.Strategy, error) {
		return &noSignalStrategy{name: "test-strat"}, nil
	}
	sess, out := newTestSession(t, fg, cfg, getStrategy)
	sess.Advisor = agent.NullProvider{Reason: "agent advisor disabled: not enabled in config"}

	if err := sess.RunOnce(context.Background(), false); err != nil {
		t.Fatalf("RunOnce() error = %v", err)
	}

	if !strings.Contains(out.String(), "advisor disabled: agent advisor disabled: not enabled in config") {
		t.Errorf("report missing the advisor-disabled one-liner: %q", out.String())
	}
}

func TestRunOnce_NilAdvisorNoAdvisorySection(t *testing.T) {
	fg := newFakeGateway(t)
	dataDir := t.TempDir()
	writeTestCSV(t, dataDir, "SPY", freshBars("SPY", 30, time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC), 450))

	cfg := baseConfig(dataDir)
	getStrategy := func(name string) (strategy.Strategy, error) {
		return &noSignalStrategy{name: "test-strat"}, nil
	}
	sess, out := newTestSession(t, fg, cfg, getStrategy)
	// sess.Advisor deliberately left nil.

	if err := sess.RunOnce(context.Background(), false); err != nil {
		t.Fatalf("RunOnce() error = %v", err)
	}
	if strings.Contains(out.String(), "AGENT ADVISORY") {
		t.Errorf("report has an AGENT ADVISORY section with a nil advisor: %q", out.String())
	}

	runs, err := sess.Store.ListPaperRuns(1)
	if err != nil {
		t.Fatalf("ListPaperRuns() error = %v", err)
	}
	if len(runs) == 1 && (runs[0].TraceJSON != "" || runs[0].AdvisoryJSON != "") {
		t.Errorf("agent columns = (%q, %q), want both empty with nil advisor and untraced strategy", runs[0].TraceJSON, runs[0].AdvisoryJSON)
	}
}
