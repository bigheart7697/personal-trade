// Package ibkr is a read-only client for the Interactive Brokers Client
// Portal (CP) Gateway's Web API: session/auth status, tickle, accounts,
// positions, and cash balance. It is session and account plumbing ONLY.
//
// Order placement is deliberately absent from this package. It arrives
// together with the paper-trading loop (docs/ROADMAP.md Phase 2/3) and,
// per CLAUDE.md cardinal rule 2, will route through risk.Manager.
// ApproveOrder like every other order path in this platform — no
// strategy, agent, or client method in this package may ever submit an
// order directly to the gateway.
//
// This client is mode-agnostic: it does not know or care whether the
// caller is in sim, paper, or live mode (see internal/config for that
// gate). Mode enforcement happens at call sites in cmd/tradeforge and
// internal/server, never here.
//
// See docs/OPTIONS.md §6 and §6a for the session model this client is
// built against: the gateway requires a manual browser login (2FA always),
// idle sessions expire in ~6 minutes without a /tickle call, and full
// sessions expire in at most ~24h regardless. TradeForge's daily-bar
// design means a short scheduled session once per trading day is enough —
// this client does not attempt to keep a session alive indefinitely.
package ibkr

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"

	"tradeforge/internal/config"
)

// ErrNotAuthenticated is returned (wrapped, so errors.Is works) whenever the
// gateway reports an unauthenticated session (HTTP 401 from any endpoint).
var ErrNotAuthenticated = errors.New("ibkr: gateway session not authenticated — open the gateway login page in a browser and sign in (see docs/OPTIONS.md §6a)")

// Client talks to a Client Portal Gateway instance over HTTPS.
type Client struct {
	BaseURL    string
	HTTPClient *http.Client
}

// NewClient builds a Client for cfg.GatewayURL.
//
// TLS note: the Client Portal Gateway serves a self-signed certificate on
// localhost by design (there is no way to get a CA-signed cert for
// 127.0.0.1/localhost, and IBKR ships it this way deliberately — this is
// documented gateway behavior, not a workaround). For that reason, and
// ONLY when the gateway URL's host is localhost or 127.0.0.1, the client's
// transport sets InsecureSkipVerify=true so the loopback-only self-signed
// cert doesn't block every request. If the configured host is anything
// else (a remote host), InsecureSkipVerify is never applied — a normal,
// fully-verifying client is built instead, so a misconfigured gateway_url
// can never silently disable TLS verification against a real remote
// endpoint.
func NewClient(cfg config.IBKRConfig) *Client {
	base := cfg.GatewayURL
	if base == "" {
		base = "https://localhost:5000/v1/api"
	}

	httpClient := &http.Client{Timeout: 15 * time.Second}
	if isLoopbackHost(base) {
		httpClient.Transport = &http.Transport{
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true}, //nolint:gosec // loopback-only gateway with a documented self-signed cert; guarded by isLoopbackHost.
		}
	}

	return &Client{
		BaseURL:    strings.TrimRight(base, "/"),
		HTTPClient: httpClient,
	}
}

// isLoopbackHost reports whether rawURL's host is localhost, 127.0.0.1, or
// ::1 (any port). ::1 matters in practice: Windows commonly resolves
// "localhost" to the IPv6 loopback first, and the gateway may be reached as
// https://[::1]:5000 — omitting it would make the real gateway fail TLS
// verification against its self-signed cert. A malformed URL is treated as
// non-loopback (fail closed: no TLS verification skip).
func isLoopbackHost(rawURL string) bool {
	u, err := url.Parse(rawURL)
	if err != nil {
		return false
	}
	host := u.Hostname() // strips [] from IPv6 literals
	return host == "localhost" || host == "127.0.0.1" || host == "::1"
}

// AuthStatus is the response body of POST /iserver/auth/status.
type AuthStatus struct {
	Authenticated bool `json:"authenticated"`
	Connected     bool `json:"connected"`
	Competing     bool `json:"competing"`
}

// AuthStatus calls POST /iserver/auth/status and reports whether the
// gateway session is authenticated, connected, and/or competing (another
// client holding the session).
func (c *Client) AuthStatus(ctx context.Context) (AuthStatus, error) {
	var out AuthStatus
	if err := c.doJSON(ctx, http.MethodPost, "/iserver/auth/status", &out); err != nil {
		return AuthStatus{}, err
	}
	return out, nil
}

// Tickle calls POST /tickle to keep the gateway session alive. Per
// docs/OPTIONS.md §6a, the gateway expects this roughly once a minute
// during an active session; callers own their own tickle loop / schedule.
func (c *Client) Tickle(ctx context.Context) error {
	return c.doJSON(ctx, http.MethodPost, "/tickle", nil)
}

// AuthStatusKeepAlive sends a best-effort /tickle keep-alive BEFORE reading
// the auth status, then returns the status. This is the method status
// readouts and the paper preflight should use; the bare AuthStatus stays for
// callers that want the raw, side-effect-free status.
//
// Rationale (found live 2026-07-16): the gateway's brokerage session goes
// DORMANT after a short idle — authenticated/connected flip to false within
// ~1-2 minutes of a successful login — and a single /tickle revives it
// immediately. Reading auth status WITHOUT a preceding tickle therefore
// reports a recoverable, logged-in session as logged-out, which false-alarms
// the dashboard badge and would abort an otherwise-fine paper session. The
// tickle is best-effort and NEVER masks the real status: if the session is
// genuinely dead (the user logged out), the tickle cannot revive it and the
// returned status stays unauthenticated, so every fail-closed gate downstream
// is preserved.
func (c *Client) AuthStatusKeepAlive(ctx context.Context) (AuthStatus, error) {
	_ = c.Tickle(ctx) // best-effort; a tickle error must not mask the real auth status
	return c.AuthStatus(ctx)
}

// Account is one brokerage account visible to the logged-in gateway user.
type Account struct {
	ID          string `json:"id"`
	AccountVan  string `json:"accountVan"`
	DisplayName string `json:"displayName"`
	// Currency is the account's BASE currency (e.g. "USD", "CAD"). The
	// paper session uses it to decide whether an FX conversion is needed
	// (non-USD accounts size in USD via ExchangeRate) and the dashboard
	// labels equity figures with it.
	Currency string `json:"currency"`
}

// Accounts calls GET /portfolio/accounts.
func (c *Client) Accounts(ctx context.Context) ([]Account, error) {
	var out []Account
	if err := c.doJSON(ctx, http.MethodGet, "/portfolio/accounts", &out); err != nil {
		return nil, err
	}
	return out, nil
}

// Position is one open position in an account's portfolio.
type Position struct {
	Conid        int64   `json:"conid"`
	ContractDesc string  `json:"contractDesc"`
	Pos          float64 `json:"position"`
	MktValue     float64 `json:"mktValue"`
	Currency     string  `json:"currency"`
}

// Positions calls GET /portfolio/{accountID}/positions/0 (page 0; IBKR
// paginates positions, and one page is enough for a personal account's
// scale).
func (c *Client) Positions(ctx context.Context, accountID string) ([]Position, error) {
	var out []Position
	path := fmt.Sprintf("/portfolio/%s/positions/0", url.PathEscape(accountID))
	if err := c.doJSON(ctx, http.MethodGet, path, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// ledgerEntry is one currency's entry within GET /portfolio/{accountID}/ledger.
type ledgerEntry struct {
	Currency            string  `json:"currency"`
	CashBalance         float64 `json:"cashbalance"`
	NetLiquidationValue float64 `json:"netliquidationvalue"`
}

// AccountTotals returns the account's cash balance AND net liquidation
// value from the ledger's BASE entry (IBKR's aggregate in the account's
// base currency), falling back to USD, then a sole entry — the same
// preference chain as CashBalance but BASE-first, because after the first
// trade a bare currency entry stops meaning "the account's money": buying
// USD securities from a CAD account creates a NEGATIVE USD cash line
// (borrowed USD) while the CAD million sits untouched — found live
// 2026-07-07, when USD-first cash made the session report $59 of equity on
// a $1M account and would have convinced the drawdown kill-switch the
// account was down 99.99%. NetLiquidationValue is the coherent
// whole-account equity figure; callers should prefer it for equity and use
// cash only as cash.
func (c *Client) AccountTotals(ctx context.Context, accountID string) (cash, netLiq float64, err error) {
	var out map[string]ledgerEntry
	path := fmt.Sprintf("/portfolio/%s/ledger", url.PathEscape(accountID))
	if err := c.doJSON(ctx, http.MethodGet, path, &out); err != nil {
		return 0, 0, err
	}
	// Only BASE's netliquidationvalue is a trustworthy WHOLE-ACCOUNT equity
	// figure. A per-currency entry's netliquidationvalue is only that
	// currency's slice, so for USD/sole fallbacks we return the cash but
	// leave netLiq 0 — the caller then falls back to marking positions
	// itself rather than mistaking a currency slice for the account total.
	if entry, ok := out["BASE"]; ok {
		return entry.CashBalance, entry.NetLiquidationValue, nil
	}
	if entry, ok := out["USD"]; ok {
		return entry.CashBalance, 0, nil
	}
	if len(out) == 1 {
		for _, entry := range out {
			return entry.CashBalance, 0, nil
		}
	}
	currencies := make([]string, 0, len(out))
	for cur := range out {
		currencies = append(currencies, cur)
	}
	sort.Strings(currencies)
	return 0, 0, fmt.Errorf("ibkr: ledger for account %s has no BASE or USD entry (currencies present: %v)", accountID, currencies)
}

// CashBalance calls GET /portfolio/{accountID}/ledger and returns the most
// useful cash figure available: the "USD" entry when present, else the
// "BASE" entry (IBKR's aggregate in the account's base currency — a fresh
// non-USD account, e.g. a CAD paper account that has never traded, has no
// USD ledger entry at all; found against the live gateway 2026-07-07), else
// the sole currency entry when exactly one exists. The value is therefore
// denominated in whichever currency won — acceptable as the sizing budget
// for paper evidence-gathering, and flagged here so a future multi-currency
// pass knows this is an approximation, not an FX-aware conversion.
func (c *Client) CashBalance(ctx context.Context, accountID string) (float64, error) {
	var out map[string]ledgerEntry
	path := fmt.Sprintf("/portfolio/%s/ledger", url.PathEscape(accountID))
	if err := c.doJSON(ctx, http.MethodGet, path, &out); err != nil {
		return 0, err
	}
	if entry, ok := out["USD"]; ok {
		return entry.CashBalance, nil
	}
	if entry, ok := out["BASE"]; ok {
		return entry.CashBalance, nil
	}
	if len(out) == 1 {
		for _, entry := range out {
			return entry.CashBalance, nil
		}
	}
	currencies := make([]string, 0, len(out))
	for cur := range out {
		currencies = append(currencies, cur)
	}
	sort.Strings(currencies)
	return 0, fmt.Errorf("ibkr: ledger for account %s has no USD or BASE entry (currencies present: %v)", accountID, currencies)
}

// ExchangeRate calls GET /iserver/exchangerate and returns the gateway's
// current FX rate from source to target: rate = target units per one source
// unit (e.g. ExchangeRate(ctx, "USD", "CAD") ≈ 1.3720 means 1 USD = 1.3720
// CAD). A non-positive rate from the gateway is rejected with an error —
// callers use this figure to size real orders (see internal/paper), and a
// zero/negative rate would silently corrupt every downstream conversion.
func (c *Client) ExchangeRate(ctx context.Context, source, target string) (float64, error) {
	var out struct {
		Rate float64 `json:"rate"`
	}
	path := fmt.Sprintf("/iserver/exchangerate?source=%s&target=%s",
		url.QueryEscape(source), url.QueryEscape(target))
	if err := c.doJSON(ctx, http.MethodGet, path, &out); err != nil {
		return 0, err
	}
	if out.Rate <= 0 {
		return 0, fmt.Errorf("ibkr: exchange rate %s->%s from the gateway is %v, want > 0 — refusing to use it for currency conversion", source, target, out.Rate)
	}
	return out.Rate, nil
}

// maxErrorBodyBytes caps how much of a non-200 response body is included in
// the returned error, so a misbehaving endpoint can't dump megabytes into a
// log line.
const maxErrorBodyBytes = 512

// maxResponseBodyBytes caps how much of ANY response body is read, guarding
// against a misbehaving endpoint streaming forever. It must be generous:
// success bodies are real payloads (/portfolio/accounts alone is ~600 bytes;
// position lists grow with holdings) — capping reads at maxErrorBodyBytes
// truncated valid JSON mid-document and broke every response over 512 bytes
// (found against the live gateway 2026-07-07; the hermetic tests all used
// smaller payloads).
const maxResponseBodyBytes = 10 << 20 // 10 MB

// doJSON issues an HTTP request to path against c.BaseURL, decoding a JSON
// response body into out (skipped if out is nil, e.g. Tickle's empty-body
// response). Non-200 responses become errors carrying the status and a
// trimmed response body; a 401 always becomes ErrNotAuthenticated
// (wrapped, so errors.Is(err, ErrNotAuthenticated) works even though the
// status/body context is preserved in the error text). Connection failures
// are wrapped with a hint that the gateway may not be running.
func (c *Client) doJSON(ctx context.Context, method, path string, out interface{}) error {
	req, err := http.NewRequestWithContext(ctx, method, c.BaseURL+path, nil)
	if err != nil {
		return fmt.Errorf("ibkr: building request for %s %s: %w", method, path, err)
	}
	req.Header.Set("Accept", "application/json")

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		if isConnRefused(err) {
			// The check matches any net.Error (refused, timeout, no route),
			// so the hint says "unreachable" rather than claiming to know
			// the gateway is down.
			return fmt.Errorf("ibkr: gateway at %s unreachable: %w (is the Client Portal Gateway running and reachable?)", c.BaseURL, err)
		}
		return fmt.Errorf("ibkr: request %s %s failed: %w", method, path, err)
	}
	defer resp.Body.Close()

	body, readErr := io.ReadAll(io.LimitReader(resp.Body, maxResponseBodyBytes))

	// readErr is deliberately checked only after the 401 and non-200
	// branches: neither needs a complete body (the 401 maps to
	// ErrNotAuthenticated regardless, and the non-200 path only quotes
	// whatever partial body it got).
	if resp.StatusCode == http.StatusUnauthorized {
		return fmt.Errorf("%w (status 401 from %s %s)", ErrNotAuthenticated, method, path)
	}

	if resp.StatusCode != http.StatusOK {
		trimmed := strings.TrimSpace(string(body))
		if len(trimmed) > maxErrorBodyBytes {
			trimmed = trimmed[:maxErrorBodyBytes]
		}
		return fmt.Errorf("ibkr: %s %s returned status %d: %s", method, path, resp.StatusCode, trimmed)
	}

	if readErr != nil {
		return fmt.Errorf("ibkr: reading response body for %s %s: %w", method, path, readErr)
	}

	if out == nil || len(strings.TrimSpace(string(body))) == 0 {
		return nil
	}

	if err := json.Unmarshal(body, out); err != nil {
		return fmt.Errorf("ibkr: decoding response for %s %s: %w", method, path, err)
	}
	return nil
}

// isConnRefused reports whether err represents a failure to even reach the
// gateway (connection refused, no such host, etc.) as opposed to an HTTP
// error response.
func isConnRefused(err error) bool {
	var netErr net.Error
	if errors.As(err, &netErr) {
		return true
	}
	var opErr *net.OpError
	return errors.As(err, &opErr)
}
