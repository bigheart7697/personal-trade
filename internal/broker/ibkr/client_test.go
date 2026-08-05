package ibkr

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"tradeforge/internal/config"
)

// newTestClient builds a Client pointed at ts, bypassing NewClient's own
// gateway_url-based TLS decision (the TLS-guard behavior itself is tested
// separately in TestNewClient_TLSGuard) so each endpoint test can focus on
// request/response handling with a plain httptest server.
func newTestClient(ts *httptest.Server) *Client {
	return &Client{
		BaseURL:    ts.URL,
		HTTPClient: ts.Client(),
	}
}

func TestAuthStatus_Happy(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/iserver/auth/status" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(AuthStatus{Authenticated: true, Connected: true, Competing: false})
	}))
	defer ts.Close()

	c := newTestClient(ts)
	status, err := c.AuthStatus(context.Background())
	if err != nil {
		t.Fatalf("AuthStatus() error = %v", err)
	}
	if !status.Authenticated || !status.Connected || status.Competing {
		t.Errorf("AuthStatus() = %+v, want {true true false}", status)
	}
}

func TestAuthStatus_Unauthenticated(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		w.Write([]byte(`{"error":"not authenticated"}`))
	}))
	defer ts.Close()

	c := newTestClient(ts)
	_, err := c.AuthStatus(context.Background())
	if err == nil {
		t.Fatal("AuthStatus() error = nil, want ErrNotAuthenticated")
	}
	if !errors.Is(err, ErrNotAuthenticated) {
		t.Errorf("AuthStatus() error = %v, want errors.Is(err, ErrNotAuthenticated)", err)
	}
}

func TestTickle_OK(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/tickle" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"session":"abc123"}`))
	}))
	defer ts.Close()

	c := newTestClient(ts)
	if err := c.Tickle(context.Background()); err != nil {
		t.Fatalf("Tickle() error = %v", err)
	}
}

func TestTickle_Unauthenticated(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer ts.Close()

	c := newTestClient(ts)
	err := c.Tickle(context.Background())
	if !errors.Is(err, ErrNotAuthenticated) {
		t.Errorf("Tickle() error = %v, want errors.Is(err, ErrNotAuthenticated)", err)
	}
}

// TestAuthStatusKeepAlive_TicklesThenReadsStatus verifies the dormant-session
// revive path (found live 2026-07-16): the keep-alive method must POST /tickle
// BEFORE it reads /iserver/auth/status, so an idle-but-recoverable brokerage
// session is woken before the status is read.
func TestAuthStatusKeepAlive_TicklesThenReadsStatus(t *testing.T) {
	var order []string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		order = append(order, r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/tickle":
			w.Write([]byte(`{}`))
		case "/iserver/auth/status":
			json.NewEncoder(w).Encode(AuthStatus{Authenticated: true, Connected: true})
		default:
			http.NotFound(w, r)
		}
	}))
	defer ts.Close()

	c := newTestClient(ts)
	got, err := c.AuthStatusKeepAlive(context.Background())
	if err != nil {
		t.Fatalf("AuthStatusKeepAlive() error = %v", err)
	}
	if !got.Authenticated || !got.Connected {
		t.Errorf("AuthStatusKeepAlive() = %+v, want authenticated+connected", got)
	}
	if len(order) != 2 || order[0] != "/tickle" || order[1] != "/iserver/auth/status" {
		t.Errorf("request order = %v, want [/tickle /iserver/auth/status]", order)
	}
}

// TestAuthStatusKeepAlive_TickleFailureDoesNotMaskStatus: a failing tickle
// must not prevent (or alter) the auth-status read — the keep-alive is
// best-effort and fail-closed semantics on the real status are preserved.
func TestAuthStatusKeepAlive_TickleFailureDoesNotMaskStatus(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/tickle":
			w.WriteHeader(http.StatusInternalServerError) // tickle fails
		case "/iserver/auth/status":
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(AuthStatus{Authenticated: true, Connected: true})
		default:
			http.NotFound(w, r)
		}
	}))
	defer ts.Close()

	c := newTestClient(ts)
	got, err := c.AuthStatusKeepAlive(context.Background())
	if err != nil {
		t.Fatalf("AuthStatusKeepAlive() error = %v, want nil (tickle error is best-effort)", err)
	}
	if !got.Authenticated {
		t.Errorf("AuthStatusKeepAlive() = %+v, want authenticated (tickle failure must not mask status)", got)
	}
}

// TestAccounts_LargeBodyNotTruncated is the regression test for the
// truncation bug found against the LIVE gateway (2026-07-07): every response
// body was read through io.LimitReader(_, maxErrorBodyBytes+1), so any
// SUCCESS payload over 512 bytes — the real /portfolio/accounts response is
// ~615 bytes — was cut mid-JSON and failed to decode. All earlier hermetic
// tests used payloads under the cap, which is exactly why they never caught
// it. This payload is deliberately far larger than maxErrorBodyBytes.
func TestAccounts_LargeBodyNotTruncated(t *testing.T) {
	// Mirror the real gateway's verbose account objects: plenty of extra
	// fields the client ignores, pushing each entry well past 512 bytes.
	entry := func(id string) string {
		return `{"id":"` + id + `","PrepaidCrypto-Z":false,"PrepaidCrypto-P":false,` +
			`"brokerageAccess":true,"accountId":"` + id + `","accountVan":"` + id + `",` +
			`"accountTitle":"Regression Fixture Account With A Deliberately Long Title",` +
			`"displayName":"Regression Fixture Account With A Deliberately Long Title",` +
			`"accountAlias":null,"accountStatus":1783310400000,"currency":"CAD",` +
			`"type":"DEMO","tradingType":"STKNOPT","faclient":false,"clearingStatus":"O",` +
			`"covestor":false,"parent":{"mmc":[],"accountId":"","isMParent":false,` +
			`"isMChild":false,"isMultiplex":false},"desc":"padding padding padding"}`
	}
	payload := "[" + entry("DUR000001") + "," + entry("DUR000002") + "]"
	if len(payload) <= maxErrorBodyBytes {
		t.Fatalf("fixture too small to exercise the regression: %d bytes (need > %d)", len(payload), maxErrorBodyBytes)
	}

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(payload))
	}))
	defer ts.Close()

	c := newTestClient(ts)
	accounts, err := c.Accounts(context.Background())
	if err != nil {
		t.Fatalf("Accounts() with a %d-byte body error = %v (truncation regression?)", len(payload), err)
	}
	if len(accounts) != 2 || accounts[0].ID != "DUR000001" || accounts[1].ID != "DUR000002" {
		t.Errorf("Accounts() = %+v, want the two fixture accounts", accounts)
	}
}

func TestAccounts_ParsesCannedPayload(t *testing.T) {
	const payload = `[
		{"id": "U1234567", "accountVan": "U1234567", "displayName": "Individual Account"},
		{"id": "U7654321", "accountVan": "U7654321", "displayName": "Paper Account"}
	]`
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/portfolio/accounts" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(payload))
	}))
	defer ts.Close()

	c := newTestClient(ts)
	accounts, err := c.Accounts(context.Background())
	if err != nil {
		t.Fatalf("Accounts() error = %v", err)
	}
	if len(accounts) != 2 {
		t.Fatalf("len(accounts) = %d, want 2", len(accounts))
	}
	if accounts[0].ID != "U1234567" || accounts[0].DisplayName != "Individual Account" {
		t.Errorf("accounts[0] = %+v, want ID=U1234567 DisplayName=\"Individual Account\"", accounts[0])
	}
	if accounts[1].DisplayName != "Paper Account" {
		t.Errorf("accounts[1].DisplayName = %q, want \"Paper Account\"", accounts[1].DisplayName)
	}
}

func TestPositions_ParsesCannedPayload(t *testing.T) {
	const payload = `[
		{"conid": 265598, "contractDesc": "AAPL", "position": 10, "mktValue": 1900.50, "currency": "USD"},
		{"conid": 12345, "contractDesc": "VXUS", "position": -5, "mktValue": -300.25, "currency": "USD"}
	]`
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		wantPath := "/portfolio/U1234567/positions/0"
		if r.Method != http.MethodGet || r.URL.Path != wantPath {
			t.Errorf("unexpected request: %s %s, want GET %s", r.Method, r.URL.Path, wantPath)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(payload))
	}))
	defer ts.Close()

	c := newTestClient(ts)
	positions, err := c.Positions(context.Background(), "U1234567")
	if err != nil {
		t.Fatalf("Positions() error = %v", err)
	}
	if len(positions) != 2 {
		t.Fatalf("len(positions) = %d, want 2", len(positions))
	}
	if positions[0].ContractDesc != "AAPL" || positions[0].Pos != 10 || positions[0].MktValue != 1900.50 {
		t.Errorf("positions[0] = %+v, want ContractDesc=AAPL Pos=10 MktValue=1900.50", positions[0])
	}
	if positions[1].Pos != -5 {
		t.Errorf("positions[1].Pos = %v, want -5", positions[1].Pos)
	}
}

// TestAccountTotals covers the BASE-first equity helper added 2026-07-07
// after a CAD-base account holding USD longs reported $59 of equity (a bare
// USD cash line goes negative post-trade). Only BASE yields a trustworthy
// whole-account netliquidationvalue.
func TestAccountTotals(t *testing.T) {
	tests := []struct {
		name       string
		payload    string
		wantCash   float64
		wantNetLiq float64
		wantErr    bool
	}{
		{
			name:       "BASE gives cash and netLiq",
			payload:    `{"BASE":{"currency":"BASE","cashbalance":716773.94,"netliquidationvalue":999985.60},"USD":{"currency":"USD","cashbalance":-199600.0,"netliquidationvalue":-100.0}}`,
			wantCash:   716773.94,
			wantNetLiq: 999985.60,
		},
		{
			name:       "USD fallback gives cash but netLiq 0 (per-currency netLiq is not the account total)",
			payload:    `{"USD":{"currency":"USD","cashbalance":54321.99,"netliquidationvalue":54321.99},"EUR":{"currency":"EUR","cashbalance":100.0}}`,
			wantCash:   54321.99,
			wantNetLiq: 0,
		},
		{
			name:       "sole non-USD entry gives cash, netLiq 0",
			payload:    `{"CAD":{"currency":"CAD","cashbalance":1000000.0,"netliquidationvalue":1000000.0}}`,
			wantCash:   1000000.0,
			wantNetLiq: 0,
		},
		{
			name:    "no BASE or USD among several currencies errors",
			payload: `{"CAD":{"currency":"CAD","cashbalance":1.0},"EUR":{"currency":"EUR","cashbalance":2.0}}`,
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.Write([]byte(tt.payload))
			}))
			defer ts.Close()

			c := newTestClient(ts)
			cash, netLiq, err := c.AccountTotals(context.Background(), "U1")
			if tt.wantErr {
				if err == nil {
					t.Fatalf("AccountTotals() error = nil, want error")
				}
				return
			}
			if err != nil {
				t.Fatalf("AccountTotals() error = %v", err)
			}
			if cash != tt.wantCash {
				t.Errorf("cash = %v, want %v", cash, tt.wantCash)
			}
			if netLiq != tt.wantNetLiq {
				t.Errorf("netLiq = %v, want %v", netLiq, tt.wantNetLiq)
			}
		})
	}
}

func TestCashBalance_ExtractsUSDFromMultiCurrencyLedger(t *testing.T) {
	const payload = `{
		"USD": {"currency": "USD", "cashbalance": 54321.99},
		"EUR": {"currency": "EUR", "cashbalance": 100.00},
		"BASE": {"currency": "BASE", "cashbalance": 54400.12}
	}`
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		wantPath := "/portfolio/U1234567/ledger"
		if r.Method != http.MethodGet || r.URL.Path != wantPath {
			t.Errorf("unexpected request: %s %s, want GET %s", r.Method, r.URL.Path, wantPath)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(payload))
	}))
	defer ts.Close()

	c := newTestClient(ts)
	cash, err := c.CashBalance(context.Background(), "U1234567")
	if err != nil {
		t.Fatalf("CashBalance() error = %v", err)
	}
	if cash != 54321.99 {
		t.Errorf("CashBalance() = %v, want 54321.99", cash)
	}
}

func TestCashBalance_FallsBackToBASEWhenNoUSD(t *testing.T) {
	// A fresh non-USD (e.g. CAD) paper account that has never traded has no
	// USD ledger entry at all — found against the live gateway 2026-07-07.
	// CashBalance must fall back to the BASE entry (IBKR's aggregate in the
	// account's base currency) rather than erroring.
	const payload = `{
		"EUR": {"currency": "EUR", "cashbalance": 100.00},
		"BASE": {"currency": "BASE", "cashbalance": 250.50}
	}`
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(payload))
	}))
	defer ts.Close()

	c := newTestClient(ts)
	cash, err := c.CashBalance(context.Background(), "U1234567")
	if err != nil {
		t.Fatalf("CashBalance() error = %v", err)
	}
	if cash != 250.50 {
		t.Errorf("CashBalance() = %v, want 250.50 (the BASE entry)", cash)
	}
}

func TestCashBalance_SingleNonUSDNonBASEEntry_ReturnsIt(t *testing.T) {
	// No USD and no BASE entry, but exactly one currency present: that
	// currency's cashbalance is the only sensible figure available, so
	// CashBalance returns it rather than erroring.
	const payload = `{"EUR": {"currency": "EUR", "cashbalance": 100.00}}`
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(payload))
	}))
	defer ts.Close()

	c := newTestClient(ts)
	cash, err := c.CashBalance(context.Background(), "U1234567")
	if err != nil {
		t.Fatalf("CashBalance() error = %v", err)
	}
	if cash != 100.00 {
		t.Errorf("CashBalance() = %v, want 100.00 (the sole EUR entry)", cash)
	}
}

func TestCashBalance_MultipleNonUSDNonBASEEntries_Errors(t *testing.T) {
	// No USD, no BASE, and MORE than one currency present: there is no
	// single sensible figure to pick, so CashBalance must error rather than
	// silently guessing a currency.
	const payload = `{
		"EUR": {"currency": "EUR", "cashbalance": 100.00},
		"GBP": {"currency": "GBP", "cashbalance": 50.00}
	}`
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(payload))
	}))
	defer ts.Close()

	c := newTestClient(ts)
	_, err := c.CashBalance(context.Background(), "U1234567")
	if err == nil {
		t.Fatal("CashBalance() error = nil, want error listing currencies for an ambiguous multi-currency ledger")
	}
	if !strings.Contains(err.Error(), "EUR") || !strings.Contains(err.Error(), "GBP") {
		t.Errorf("CashBalance() error = %q, want it to list both currencies present (EUR, GBP)", err.Error())
	}
}

func TestExchangeRate_Happy(t *testing.T) {
	var gotSource, gotTarget string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/iserver/exchangerate" {
			t.Errorf("unexpected request: %s %s, want GET /iserver/exchangerate", r.Method, r.URL.Path)
		}
		gotSource = r.URL.Query().Get("source")
		gotTarget = r.URL.Query().Get("target")
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"rate": 1.3720}`))
	}))
	defer ts.Close()

	c := newTestClient(ts)
	rate, err := c.ExchangeRate(context.Background(), "USD", "CAD")
	if err != nil {
		t.Fatalf("ExchangeRate() error = %v", err)
	}
	if rate != 1.3720 {
		t.Errorf("ExchangeRate() = %v, want 1.3720", rate)
	}
	if gotSource != "USD" || gotTarget != "CAD" {
		t.Errorf("query params source=%q target=%q, want source=USD target=CAD", gotSource, gotTarget)
	}
}

func TestExchangeRate_NonPositiveRateErrors(t *testing.T) {
	// A zero or negative rate would silently corrupt every downstream
	// currency conversion (internal/paper sizes real orders with it), so the
	// client must reject it rather than return it.
	for _, payload := range []string{`{"rate": 0}`, `{"rate": -1.25}`, `{}`} {
		t.Run(payload, func(t *testing.T) {
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.Write([]byte(payload))
			}))
			defer ts.Close()

			c := newTestClient(ts)
			_, err := c.ExchangeRate(context.Background(), "USD", "CAD")
			if err == nil {
				t.Fatalf("ExchangeRate() with body %s error = nil, want a non-positive-rate error", payload)
			}
			if !strings.Contains(err.Error(), "want > 0") {
				t.Errorf("ExchangeRate() error = %q, want it to explain the rate must be > 0", err.Error())
			}
		})
	}
}

func TestExchangeRate_Unauthenticated(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer ts.Close()

	c := newTestClient(ts)
	_, err := c.ExchangeRate(context.Background(), "USD", "CAD")
	if !errors.Is(err, ErrNotAuthenticated) {
		t.Errorf("ExchangeRate() error = %v, want errors.Is(err, ErrNotAuthenticated)", err)
	}
}

func TestNon200_ErrorIncludesStatus(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("internal gateway error"))
	}))
	defer ts.Close()

	c := newTestClient(ts)
	_, err := c.AuthStatus(context.Background())
	if err == nil {
		t.Fatal("AuthStatus() error = nil, want error for 500")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("AuthStatus() error = %q, want it to mention status 500", err.Error())
	}
	if !strings.Contains(err.Error(), "internal gateway error") {
		t.Errorf("AuthStatus() error = %q, want it to include the response body", err.Error())
	}
}

func TestNon200_ErrorBodyCappedAt512Bytes(t *testing.T) {
	longBody := strings.Repeat("x", 2000)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		w.Write([]byte(longBody))
	}))
	defer ts.Close()

	c := newTestClient(ts)
	_, err := c.AuthStatus(context.Background())
	if err == nil {
		t.Fatal("AuthStatus() error = nil, want error for 502")
	}
	// The error text includes other prefix content, so just verify it does
	// not embed the full 2000-char body verbatim.
	if strings.Contains(err.Error(), longBody) {
		t.Errorf("AuthStatus() error embeds the full uncapped body (len %d)", len(longBody))
	}
}

func TestNewClient_TLSGuard(t *testing.T) {
	t.Run("localhost gets InsecureSkipVerify", func(t *testing.T) {
		c := NewClient(config.IBKRConfig{GatewayURL: "https://localhost:5000/v1/api"})
		transport, ok := c.HTTPClient.Transport.(*http.Transport)
		if !ok || transport == nil {
			t.Fatalf("expected a *http.Transport for a localhost URL, got %T", c.HTTPClient.Transport)
		}
		if transport.TLSClientConfig == nil || !transport.TLSClientConfig.InsecureSkipVerify {
			t.Error("expected InsecureSkipVerify=true for a localhost gateway URL")
		}
	})

	t.Run("127.0.0.1 gets InsecureSkipVerify", func(t *testing.T) {
		c := NewClient(config.IBKRConfig{GatewayURL: "https://127.0.0.1:5000/v1/api"})
		transport, ok := c.HTTPClient.Transport.(*http.Transport)
		if !ok || transport == nil {
			t.Fatalf("expected a *http.Transport for a 127.0.0.1 URL, got %T", c.HTTPClient.Transport)
		}
		if transport.TLSClientConfig == nil || !transport.TLSClientConfig.InsecureSkipVerify {
			t.Error("expected InsecureSkipVerify=true for a 127.0.0.1 gateway URL")
		}
	})

	t.Run("IPv6 loopback [::1] gets InsecureSkipVerify", func(t *testing.T) {
		// Windows commonly resolves localhost to ::1 first; the gateway must
		// stay reachable when addressed via the IPv6 loopback literal.
		c := NewClient(config.IBKRConfig{GatewayURL: "https://[::1]:5000/v1/api"})
		transport, ok := c.HTTPClient.Transport.(*http.Transport)
		if !ok || transport == nil {
			t.Fatalf("expected a *http.Transport for a [::1] URL, got %T", c.HTTPClient.Transport)
		}
		if transport.TLSClientConfig == nil || !transport.TLSClientConfig.InsecureSkipVerify {
			t.Error("expected InsecureSkipVerify=true for a [::1] gateway URL")
		}
	})

	t.Run("remote host does NOT get InsecureSkipVerify", func(t *testing.T) {
		c := NewClient(config.IBKRConfig{GatewayURL: "https://gateway.example.com:5000/v1/api"})
		// A normal client must NOT skip verification for a remote host: no
		// custom Transport carrying InsecureSkipVerify=true should be set.
		if transport, ok := c.HTTPClient.Transport.(*http.Transport); ok && transport != nil {
			if transport.TLSClientConfig != nil && transport.TLSClientConfig.InsecureSkipVerify {
				t.Error("InsecureSkipVerify=true must never be set for a non-loopback host")
			}
		}
	})

	t.Run("default gateway URL used when empty", func(t *testing.T) {
		c := NewClient(config.IBKRConfig{})
		if c.BaseURL != "https://localhost:5000/v1/api" {
			t.Errorf("BaseURL = %q, want default", c.BaseURL)
		}
		transport, ok := c.HTTPClient.Transport.(*http.Transport)
		if !ok || transport == nil || transport.TLSClientConfig == nil || !transport.TLSClientConfig.InsecureSkipVerify {
			t.Error("expected InsecureSkipVerify=true for the default (localhost) gateway URL")
		}
	})
}
