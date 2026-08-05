package ibkr

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func TestLookupConid_Happy(t *testing.T) {
	const payload = `[
		{"conid": "265598", "symbol": "AAPL", "sections": [{"secType": "STK"}, {"secType": "OPT"}]}
	]`
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/iserver/secdef/search" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		var body map[string]string
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decoding request body: %v", err)
		}
		if body["symbol"] != "AAPL" {
			t.Errorf("request symbol = %q, want AAPL", body["symbol"])
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(payload))
	}))
	defer ts.Close()

	c := newTestClient(ts)
	conid, err := c.LookupConid(context.Background(), "AAPL")
	if err != nil {
		t.Fatalf("LookupConid() error = %v", err)
	}
	if conid != 265598 {
		t.Errorf("LookupConid() = %d, want 265598", conid)
	}
}

func TestLookupConid_Ambiguous(t *testing.T) {
	const payload = `[
		{"conid": "111", "symbol": "AAPL", "sections": [{"secType": "STK"}]},
		{"conid": "222", "symbol": "AAPL", "sections": [{"secType": "STK"}]}
	]`
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(payload))
	}))
	defer ts.Close()

	c := newTestClient(ts)
	_, err := c.LookupConid(context.Background(), "AAPL")
	if err == nil {
		t.Fatal("LookupConid() error = nil, want ambiguous error")
	}
	if !strings.Contains(err.Error(), "ambiguous") {
		t.Errorf("LookupConid() error = %q, want it to mention ambiguity", err.Error())
	}
}

// TestLookupConid_USPrimaryDisambiguation mirrors the LIVE payload that
// broke the first real order (2026-07-07): "QQQ" matches the Invesco NASDAQ
// ETF, its Mexican listing (MEXI), and Questcorp Mining (PURE) — three STK
// contracts, two of them the wrong thing to buy. The US-primary-listing
// filter must pick the NASDAQ contract and only that one.
func TestLookupConid_USPrimaryDisambiguation(t *testing.T) {
	const payload = `[
		{"conid": "320227571", "symbol": "QQQ", "description": "NASDAQ", "sections": [{"secType": "STK"}, {"secType": "OPT"}]},
		{"conid": "320227574", "symbol": "QQQ", "description": "MEXI", "sections": [{"secType": "STK"}]},
		{"conid": "705592547", "symbol": "QQQ", "description": "PURE", "sections": [{"secType": "STK"}]}
	]`
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(payload))
	}))
	defer ts.Close()

	c := newTestClient(ts)
	conid, err := c.LookupConid(context.Background(), "QQQ")
	if err != nil {
		t.Fatalf("LookupConid() error = %v, want the NASDAQ contract", err)
	}
	if conid != 320227571 {
		t.Errorf("LookupConid() = %d, want 320227571 (the NASDAQ listing)", conid)
	}
}

// TestLookupConid_TwoUSListingsStillAmbiguous: if US-primary filtering
// leaves more than one candidate, the lookup must still fail closed and
// name every candidate.
func TestLookupConid_TwoUSListingsStillAmbiguous(t *testing.T) {
	const payload = `[
		{"conid": "111", "symbol": "DUP", "description": "NASDAQ", "sections": [{"secType": "STK"}]},
		{"conid": "222", "symbol": "DUP", "description": "NYSE", "sections": [{"secType": "STK"}]}
	]`
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(payload))
	}))
	defer ts.Close()

	c := newTestClient(ts)
	_, err := c.LookupConid(context.Background(), "DUP")
	if err == nil {
		t.Fatal("LookupConid() error = nil, want ambiguity error")
	}
	if !strings.Contains(err.Error(), "NASDAQ") || !strings.Contains(err.Error(), "NYSE") {
		t.Errorf("LookupConid() error = %q, want both candidates named", err.Error())
	}
}

func TestLookupConid_NoMatch(t *testing.T) {
	tests := []struct {
		name    string
		payload string
	}{
		{"empty results", `[]`},
		{"symbol mismatch", `[{"conid": "1", "symbol": "AAPLX", "sections": [{"secType": "STK"}]}]`},
		{"no STK section", `[{"conid": "1", "symbol": "AAPL", "sections": [{"secType": "OPT"}]}]`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.Write([]byte(tt.payload))
			}))
			defer ts.Close()

			c := newTestClient(ts)
			_, err := c.LookupConid(context.Background(), "AAPL")
			if err == nil {
				t.Fatal("LookupConid() error = nil, want no-match error")
			}
		})
	}
}

func TestPlaceOrder_NoQuestions(t *testing.T) {
	var gotBody map[string]any
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		wantPath := "/iserver/account/U123/orders"
		if r.Method != http.MethodPost || r.URL.Path != wantPath {
			t.Errorf("unexpected request: %s %s, want POST %s", r.Method, r.URL.Path, wantPath)
		}
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Fatalf("decoding request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`[{"order_id": "999", "order_status": "Submitted"}]`))
	}))
	defer ts.Close()

	c := newTestClient(ts)
	result, err := c.PlaceOrder(context.Background(), OrderRequest{
		AccountID: "U123",
		Conid:     265598,
		Side:      "BUY",
		Quantity:  10,
		OrderType: "MKT",
		TIF:       "DAY",
		COID:      "tf-2026-07-06-test-AAPL-1",
	})
	if err != nil {
		t.Fatalf("PlaceOrder() error = %v", err)
	}
	if result.OrderID != "999" || result.Status != "Submitted" {
		t.Errorf("PlaceOrder() = %+v, want {999 Submitted}", result)
	}

	orders, ok := gotBody["orders"].([]any)
	if !ok || len(orders) != 1 {
		t.Fatalf("request body orders = %v, want a 1-element array", gotBody["orders"])
	}
	order := orders[0].(map[string]any)
	if order["cOID"] != "tf-2026-07-06-test-AAPL-1" {
		t.Errorf("request cOID = %v, want tf-2026-07-06-test-AAPL-1", order["cOID"])
	}
}

func TestPlaceOrder_AllowlistedQuestion_RepliesAndSucceeds(t *testing.T) {
	var callCount int32
	var replyCalled int32
	var replyPath string

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case strings.HasPrefix(r.URL.Path, "/iserver/account/") && strings.HasSuffix(r.URL.Path, "/orders"):
			atomic.AddInt32(&callCount, 1)
			w.Write([]byte(`[{"id": "q1", "message": ["Order size exceeds the size limit you set."]}]`))
		case strings.HasPrefix(r.URL.Path, "/iserver/reply/"):
			atomic.AddInt32(&replyCalled, 1)
			replyPath = r.URL.Path
			var body map[string]bool
			json.NewDecoder(r.Body).Decode(&body)
			if !body["confirmed"] {
				t.Errorf("reply body confirmed = %v, want true", body["confirmed"])
			}
			w.Write([]byte(`[{"order_id": "1000", "order_status": "Submitted"}]`))
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
	}))
	defer ts.Close()

	c := newTestClient(ts)
	result, err := c.PlaceOrder(context.Background(), OrderRequest{
		AccountID: "U123",
		Conid:     265598,
		Side:      "BUY",
		Quantity:  10,
		OrderType: "MKT",
		TIF:       "DAY",
		COID:      "tf-2026-07-06-test-AAPL-2",
	})
	if err != nil {
		t.Fatalf("PlaceOrder() error = %v", err)
	}
	if result.OrderID != "1000" {
		t.Errorf("PlaceOrder() = %+v, want OrderID 1000", result)
	}
	if atomic.LoadInt32(&replyCalled) != 1 {
		t.Errorf("reply endpoint called %d times, want 1", replyCalled)
	}
	if replyPath != "/iserver/reply/q1" {
		t.Errorf("reply path = %q, want /iserver/reply/q1", replyPath)
	}
}

func TestPlaceOrder_UnknownQuestion_AbortsWithoutReply(t *testing.T) {
	var replyCalled int32

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case strings.HasPrefix(r.URL.Path, "/iserver/account/") && strings.HasSuffix(r.URL.Path, "/orders"):
			w.Write([]byte(`[{"id": "q1", "message": ["This order will exceed your margin requirements."]}]`))
		case strings.HasPrefix(r.URL.Path, "/iserver/reply/"):
			atomic.AddInt32(&replyCalled, 1)
			w.Write([]byte(`[{"order_id": "1000", "order_status": "Submitted"}]`))
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
	}))
	defer ts.Close()

	c := newTestClient(ts)
	_, err := c.PlaceOrder(context.Background(), OrderRequest{
		AccountID: "U123",
		Conid:     265598,
		Side:      "BUY",
		Quantity:  10,
		OrderType: "MKT",
		TIF:       "DAY",
		COID:      "tf-2026-07-06-test-AAPL-3",
	})
	if err == nil {
		t.Fatal("PlaceOrder() error = nil, want error for unrecognized question")
	}
	if !strings.Contains(err.Error(), "margin requirements") {
		t.Errorf("PlaceOrder() error = %q, want it to quote the question", err.Error())
	}
	if atomic.LoadInt32(&replyCalled) != 0 {
		t.Errorf("reply endpoint called %d times, want 0 (unknown question must not be confirmed)", replyCalled)
	}
}

func TestPlaceOrder_QuestionLoopCap(t *testing.T) {
	var callCount int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case strings.HasPrefix(r.URL.Path, "/iserver/account/") && strings.HasSuffix(r.URL.Path, "/orders"):
			atomic.AddInt32(&callCount, 1)
			w.Write([]byte(`[{"id": "qA", "message": ["price exceeds the current market price"]}]`))
		case strings.HasPrefix(r.URL.Path, "/iserver/reply/"):
			atomic.AddInt32(&callCount, 1)
			// Always answers with another allowlisted question, never resolving.
			w.Write([]byte(`[{"id": "qA", "message": ["price exceeds the current market price"]}]`))
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
	}))
	defer ts.Close()

	c := newTestClient(ts)
	_, err := c.PlaceOrder(context.Background(), OrderRequest{
		AccountID: "U123",
		Conid:     265598,
		Side:      "BUY",
		Quantity:  10,
		OrderType: "MKT",
		TIF:       "DAY",
		COID:      "tf-2026-07-06-test-AAPL-4",
	})
	if err == nil {
		t.Fatal("PlaceOrder() error = nil, want error for question loop cap")
	}
	if !strings.Contains(err.Error(), "did not resolve within") {
		t.Errorf("PlaceOrder() error = %q, want it to mention the round cap", err.Error())
	}
	// maxQuestionRounds = 3: the initial place-order call + up to 3
	// question rounds = at most 4 total calls hitting either endpoint.
	if got := atomic.LoadInt32(&callCount); got > 4 {
		t.Errorf("total calls = %d, want <= 4 (cap enforced)", got)
	}
}

func TestLiveOrders_ParsesCannedPayload(t *testing.T) {
	const payload = `{"orders": [
		{"orderId": "1000", "cOID": "tf-2026-07-06-test-AAPL-1", "conid": 265598, "side": "BUY", "totalSize": 10, "filledQuantity": 10, "avgPrice": 150.25, "status": "Filled"},
		{"orderId": "1001", "cOID": "tf-2026-07-06-test-MSFT-1", "conid": 272093, "side": "SELL", "totalSize": 5, "filledQuantity": 0, "avgPrice": 0, "status": "Submitted"}
	]}`
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/iserver/account/orders" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(payload))
	}))
	defer ts.Close()

	c := newTestClient(ts)
	orders, err := c.LiveOrders(context.Background())
	if err != nil {
		t.Fatalf("LiveOrders() error = %v", err)
	}
	if len(orders) != 2 {
		t.Fatalf("len(orders) = %d, want 2", len(orders))
	}
	if orders[0].OrderID != "1000" || orders[0].Status != "Filled" || orders[0].FilledQuantity != 10 || orders[0].AvgPrice != 150.25 {
		t.Errorf("orders[0] = %+v, want OrderID=1000 Status=Filled FilledQuantity=10 AvgPrice=150.25", orders[0])
	}
	if orders[1].COID != "tf-2026-07-06-test-MSFT-1" || orders[1].Side != "SELL" {
		t.Errorf("orders[1] = %+v, want COID=tf-2026-07-06-test-MSFT-1 Side=SELL", orders[1])
	}
}
