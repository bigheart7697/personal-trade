package ibkr

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
)

// allowlistedQuestionSubstrings is the ALLOWLIST of confirmation-question
// message fragments the paper-trading loop may answer "yes" to
// automatically, matched case-insensitively as a substring anywhere in the
// question's message text. Every entry here has been reviewed as benign for
// an automated paper-trading session (routine warnings the gateway asks on
// essentially every order, not risk-relevant confirmations). ANY OTHER
// confirmation question — margin, restricted-list, short-sale, "order value
// exceeds", etc. — is treated as unknown and aborts the order rather than
// being auto-confirmed.
//
// Adding to this list requires human review: it is the one place an
// automated order path can silently say "yes" to the gateway, so treat any
// addition with the same care as a change to internal/risk. Do not add
// broad or vague fragments.
var allowlistedQuestionSubstrings = []string{
	"price exceeds",
	"size limit",
	"cash quantity",
	"mixed allocation",
}

// OrderRequest is one order to place via PlaceOrder. OrderType is always
// "MKT" in v1 (limit/stop order types are not yet supported by the paper
// loop). COID is a client-assigned order id used for idempotency — the
// gateway deduplicates repeated submissions carrying the same cOId within
// its dedup window, so callers should derive it deterministically (see
// internal/paper's "tf-<date>-<strategy>-<symbol>-<seq>" scheme) rather than
// randomly, so retrying a failed request is safe.
type OrderRequest struct {
	AccountID string
	Conid     int64
	Side      string // "BUY" | "SELL"
	Quantity  int64
	OrderType string // always "MKT" in v1
	TIF       string // "DAY"
	COID      string // client order id, for idempotency
}

// OrderResult is the final, confirmed outcome of a PlaceOrder call.
type OrderResult struct {
	OrderID string
	Status  string
}

// conidSearchResult is one element of POST /iserver/secdef/search's response
// array.
type conidSearchResult struct {
	Conid    string             `json:"conid"`
	Symbol   string             `json:"symbol"`
	Sections []conidSearchEntry `json:"sections"`
	// Description carries the primary listing exchange for STK results
	// (e.g. "NASDAQ", "MEXI", "PURE") — the disambiguator when one ticker
	// matches several listings or even different companies (found live
	// 2026-07-07: "QQQ" returns the Invesco NASDAQ ETF, its Mexican
	// listing, AND Questcorp Mining on PURE).
	Description string `json:"description"`
}

// conidSearchEntry is one entry of a search result's "sections" array; each
// section names a security type (secType) available for that conid (e.g.
// "STK", "OPT", "FUT").
type conidSearchEntry struct {
	SecType string `json:"secType"`
}

// LookupConid resolves symbol to an IBKR contract id via POST
// /iserver/secdef/search, restricted to results whose symbol matches
// exactly (case-sensitive, as IBKR's own symbology is) and which have a
// "STK" section (an equity/ETF listing, not an option chain or future).
// Zero matching results, or more than one, is an error naming the ambiguity
// (or its absence) explicitly rather than guessing.
func (c *Client) LookupConid(ctx context.Context, symbol string) (int64, error) {
	body, err := json.Marshal(map[string]string{"symbol": symbol})
	if err != nil {
		return 0, fmt.Errorf("ibkr: marshal conid search request for %s: %w", symbol, err)
	}

	var results []conidSearchResult
	if err := c.doJSONBody(ctx, http.MethodPost, "/iserver/secdef/search", body, &results); err != nil {
		return 0, fmt.Errorf("ibkr: conid search for %s: %w", symbol, err)
	}

	var matches []conidSearchResult
	for _, r := range results {
		if r.Symbol != symbol {
			continue
		}
		for _, sec := range r.Sections {
			if sec.SecType == "STK" {
				matches = append(matches, r)
				break
			}
		}
	}

	switch len(matches) {
	case 0:
		return 0, fmt.Errorf("ibkr: no STK contract found for symbol %q (searched %d result(s))", symbol, len(results))
	case 1:
		conid, err := parseConid(matches[0].Conid)
		if err != nil {
			return 0, fmt.Errorf("ibkr: parsing conid for %s: %w", symbol, err)
		}
		return conid, nil
	default:
		// Same-ticker collisions across exchanges (and across COMPANIES) are
		// normal: prefer the single match whose primary listing is a US
		// exchange. If that still leaves zero or several, fail closed with
		// every candidate listed rather than guessing — buying the wrong
		// contract is far worse than not buying.
		var usMatches []conidSearchResult
		for _, m := range matches {
			if usPrimaryExchanges[m.Description] {
				usMatches = append(usMatches, m)
			}
		}
		if len(usMatches) == 1 {
			conid, err := parseConid(usMatches[0].Conid)
			if err != nil {
				return 0, fmt.Errorf("ibkr: parsing conid for %s: %w", symbol, err)
			}
			return conid, nil
		}
		candidates := make([]string, 0, len(matches))
		for _, m := range matches {
			candidates = append(candidates, fmt.Sprintf("%s (%s, conid %s)", m.Symbol, m.Description, m.Conid))
		}
		return 0, fmt.Errorf("ibkr: ambiguous symbol %q: %d STK contracts matched and US-primary-listing filtering left %d: %s",
			symbol, len(matches), len(usMatches), strings.Join(candidates, "; "))
	}
}

// usPrimaryExchanges is the set of `description` values (primary listing
// exchanges) accepted as "the US listing" when a ticker search returns
// multiple STK contracts. Additions require care: the description must
// uniquely identify a US primary listing, or LookupConid could silently
// pick the wrong contract.
var usPrimaryExchanges = map[string]bool{
	"NASDAQ": true,
	"NYSE":   true,
	"ARCA":   true,
	"AMEX":   true,
	"BATS":   true,
	"IEX":    true,
}

func parseConid(s string) (int64, error) {
	var conid int64
	if _, err := fmt.Sscanf(s, "%d", &conid); err != nil {
		return 0, fmt.Errorf("invalid conid %q: %w", s, err)
	}
	return conid, nil
}

// orderSubmission is the shape of one order in POST
// /iserver/account/{accountId}/orders's request body.
type orderSubmission struct {
	Conid     int64  `json:"conid"`
	Side      string `json:"side"`
	Quantity  int64  `json:"quantity"`
	OrderType string `json:"orderType"`
	TIF       string `json:"tif"`
	COID      string `json:"cOID"`
}

// orderReplyEntry is one element of the gateway's response to placing an
// order or replying to a confirmation question: either a terminal order
// result (OrderID/Status populated) or a pending confirmation question (ID/
// Message populated). The gateway's API reuses the same array shape for
// both, distinguished by which fields are present.
type orderReplyEntry struct {
	OrderID string   `json:"order_id"`
	Status  string   `json:"order_status"`
	ID      string   `json:"id"`
	Message []string `json:"message"`
}

// maxQuestionRounds caps how many confirmation-question round-trips
// PlaceOrder will answer before giving up; a well-behaved gateway resolves
// within one or two rounds, so hitting this cap indicates something is
// looping rather than converging and should surface as an error, not spin
// forever.
const maxQuestionRounds = 3

// PlaceOrder submits req via POST /iserver/account/{accountId}/orders. If
// the gateway responds with confirmation questions, each question's message
// lines are checked against allowlistedQuestionSubstrings (case-insensitive
// substring match); if every line of every question matches, PlaceOrder
// POSTs /iserver/reply/{id} {"confirmed":true} and re-reads the response,
// looping up to maxQuestionRounds times. The FIRST unrecognized message
// aborts immediately with an error that quotes the question verbatim and
// sends NO reply for it (fail closed — an unknown question must never be
// silently confirmed).
func (c *Client) PlaceOrder(ctx context.Context, req OrderRequest) (OrderResult, error) {
	body, err := json.Marshal(map[string]any{
		"orders": []orderSubmission{{
			Conid:     req.Conid,
			Side:      req.Side,
			Quantity:  req.Quantity,
			OrderType: req.OrderType,
			TIF:       req.TIF,
			COID:      req.COID,
		}},
	})
	if err != nil {
		return OrderResult{}, fmt.Errorf("ibkr: marshal order request: %w", err)
	}

	path := fmt.Sprintf("/iserver/account/%s/orders", url.PathEscape(req.AccountID))

	var entries []orderReplyEntry
	if err := c.doJSONBody(ctx, http.MethodPost, path, body, &entries); err != nil {
		return OrderResult{}, fmt.Errorf("ibkr: place order for %s: %w", req.COID, err)
	}

	for round := 0; round < maxQuestionRounds; round++ {
		if len(entries) == 0 {
			return OrderResult{}, fmt.Errorf("ibkr: place order for %s: gateway returned no order entries", req.COID)
		}

		entry := entries[0]

		// A terminal result carries an order id; a pending confirmation
		// carries a question id and message instead.
		if entry.OrderID != "" {
			return OrderResult{OrderID: entry.OrderID, Status: entry.Status}, nil
		}

		if entry.ID == "" {
			return OrderResult{}, fmt.Errorf("ibkr: place order for %s: gateway response has neither order_id nor a confirmation id: %+v", req.COID, entry)
		}

		if !allMessagesAllowlisted(entry.Message) {
			return OrderResult{}, fmt.Errorf(
				"ibkr: order for %s blocked by an unrecognized confirmation question (order NOT confirmed): %q",
				req.COID, strings.Join(entry.Message, " | "))
		}

		replyBody, err := json.Marshal(map[string]bool{"confirmed": true})
		if err != nil {
			return OrderResult{}, fmt.Errorf("ibkr: marshal confirmation reply for %s: %w", req.COID, err)
		}

		replyPath := fmt.Sprintf("/iserver/reply/%s", url.PathEscape(entry.ID))
		entries = nil
		if err := c.doJSONBody(ctx, http.MethodPost, replyPath, replyBody, &entries); err != nil {
			return OrderResult{}, fmt.Errorf("ibkr: confirming question %s for order %s: %w", entry.ID, req.COID, err)
		}
	}

	return OrderResult{}, fmt.Errorf("ibkr: order for %s did not resolve within %d confirmation round(s)", req.COID, maxQuestionRounds)
}

// allMessagesAllowlisted reports whether every line in messages matches at
// least one entry in allowlistedQuestionSubstrings (case-insensitive
// substring match). An empty messages slice is treated as NOT allowlisted —
// a question with no readable text is never auto-confirmed.
func allMessagesAllowlisted(messages []string) bool {
	if len(messages) == 0 {
		return false
	}
	for _, msg := range messages {
		if !isAllowlistedMessage(msg) {
			return false
		}
	}
	return true
}

func isAllowlistedMessage(msg string) bool {
	lower := strings.ToLower(msg)
	for _, allowed := range allowlistedQuestionSubstrings {
		if strings.Contains(lower, allowed) {
			return true
		}
	}
	return false
}

// LiveOrder is one order as reported by GET /iserver/account/orders.
type LiveOrder struct {
	OrderID        string
	COID           string
	Conid          int64
	Side           string
	Quantity       int64
	FilledQuantity float64
	AvgPrice       float64
	Status         string
}

// liveOrdersResponse is the shape of GET /iserver/account/orders' body: the
// gateway wraps the order array in an object.
type liveOrdersResponse struct {
	Orders []liveOrderEntry `json:"orders"`
}

// liveOrderEntry is one element of liveOrdersResponse.Orders. The gateway's
// numeric fields are documented to sometimes arrive as either JSON numbers
// or strings depending on endpoint/version; orderIdField/conidField below
// are decoded leniently to tolerate either.
type liveOrderEntry struct {
	OrderID        json.Number `json:"orderId"`
	COID           string      `json:"cOID"`
	Conid          json.Number `json:"conid"`
	Side           string      `json:"side"`
	Quantity       json.Number `json:"totalSize"`
	FilledQuantity json.Number `json:"filledQuantity"`
	AvgPrice       json.Number `json:"avgPrice"`
	Status         string      `json:"status"`
}

// LiveOrders calls GET /iserver/account/orders and returns the currently
// known orders (any status) for the logged-in gateway session.
func (c *Client) LiveOrders(ctx context.Context) ([]LiveOrder, error) {
	var resp liveOrdersResponse
	if err := c.doJSON(ctx, http.MethodGet, "/iserver/account/orders", &resp); err != nil {
		return nil, fmt.Errorf("ibkr: list live orders: %w", err)
	}

	out := make([]LiveOrder, 0, len(resp.Orders))
	for _, e := range resp.Orders {
		conid, _ := e.Conid.Int64()
		qty, _ := e.Quantity.Float64()
		filled, _ := e.FilledQuantity.Float64()
		avgPrice, _ := e.AvgPrice.Float64()

		out = append(out, LiveOrder{
			OrderID:        e.OrderID.String(),
			COID:           e.COID,
			Conid:          conid,
			Side:           e.Side,
			Quantity:       int64(qty),
			FilledQuantity: filled,
			AvgPrice:       avgPrice,
			Status:         e.Status,
		})
	}
	return out, nil
}

// doJSONBody issues an HTTP request carrying a JSON request body to path
// against c.BaseURL, decoding a JSON response body into out exactly like
// doJSON. It exists alongside doJSON (which never sends a body) because
// LookupConid/PlaceOrder/reply-confirmation all POST a JSON payload, while
// every other endpoint in this package sends none.
func (c *Client) doJSONBody(ctx context.Context, method, path string, body []byte, out interface{}) error {
	req, err := http.NewRequestWithContext(ctx, method, c.BaseURL+path, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("ibkr: building request for %s %s: %w", method, path, err)
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		if isConnRefused(err) {
			return fmt.Errorf("ibkr: gateway at %s unreachable: %w (is the Client Portal Gateway running and reachable?)", c.BaseURL, err)
		}
		return fmt.Errorf("ibkr: request %s %s failed: %w", method, path, err)
	}
	defer resp.Body.Close()

	// maxResponseBodyBytes, NOT maxErrorBodyBytes: the error cap is only for
	// trimming error-message text below — capping the read itself truncated
	// valid success JSON over 512 bytes (see the constant's doc in client.go).
	respBody, readErr := io.ReadAll(io.LimitReader(resp.Body, maxResponseBodyBytes))

	if resp.StatusCode == http.StatusUnauthorized {
		return fmt.Errorf("%w (status 401 from %s %s)", ErrNotAuthenticated, method, path)
	}

	if resp.StatusCode != http.StatusOK {
		trimmed := strings.TrimSpace(string(respBody))
		if len(trimmed) > maxErrorBodyBytes {
			trimmed = trimmed[:maxErrorBodyBytes]
		}
		return fmt.Errorf("ibkr: %s %s returned status %d: %s", method, path, resp.StatusCode, trimmed)
	}

	if readErr != nil {
		return fmt.Errorf("ibkr: reading response body for %s %s: %w", method, path, readErr)
	}

	if out == nil || len(strings.TrimSpace(string(respBody))) == 0 {
		return nil
	}

	if err := json.Unmarshal(respBody, out); err != nil {
		return fmt.Errorf("ibkr: decoding response for %s %s: %w", method, path, err)
	}
	return nil
}
