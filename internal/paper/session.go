// Package paper implements the paper-trading daily session: the code path
// where a promoted strategy's signals become real orders against IBKR's
// PAPER account. See docs/OPTIONS.md §6a for why a short scheduled session
// once per trading day is the right design for a daily-bar platform (no
// need to babysit a 24/7 gateway session).
//
// This package is mode-gated to ModePaper only (CLAUDE.md cardinal rule 1):
// ModeSim refuses with instructions to set mode "paper", and ModeLive
// refuses outright — there is no live order path in this codebase, full
// stop, and this package must never grow one. Every signal this package
// produces is sized and approved by risk.Manager.ApproveOrder (cardinal
// rule 2) before it becomes an OrderRequest; there is no other path from a
// domain.Signal to ibkr.Client.PlaceOrder.
package paper

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"
	"time"

	"tradeforge/internal/agent"
	"tradeforge/internal/broker/ibkr"
	"tradeforge/internal/config"
	"tradeforge/internal/data"
	"tradeforge/internal/domain"
	"tradeforge/internal/risk"
	"tradeforge/internal/store"
	"tradeforge/internal/strategy"
)

// staleAfter is how old (in calendar days) the newest bar of a strategy's
// data may be before RunOnce warns that it looks stale (the CSV probably
// needs a fresh `data import`/`data fetch`). This is a warning, not a
// refusal — the session still runs on whatever data it has, loudly flagged.
const staleAfter = 5 * 24 * time.Hour

// liveOrderPollTimeout bounds how long RunOnce polls LiveOrders for a fill
// status after placing an order in --execute mode.
const liveOrderPollTimeout = 30 * time.Second

// liveOrderPollInterval is the spacing between LiveOrders polls.
const liveOrderPollInterval = 2 * time.Second

// advisorTimeout bounds the (optional, advisory-only) LLM review call. The
// provider applies its own tighter internal timeout; this is the session's
// outer bound so a misbehaving provider can never stall a paper session
// indefinitely — and an advisor failure of ANY kind never fails the session
// (see the AGENT ADVISORY step in RunOnce).
const advisorTimeout = 60 * time.Second

// advisorRecentSessions is how many prior paper sessions are summarized into
// the advisor's ReviewInput.
const advisorRecentSessions = 5

// Session runs one daily paper-trading cycle: load each promoted strategy's
// data, compute its signal(s) at the last completed bar against the real
// IBKR paper portfolio, risk-approve, and (dry-run or execute) place orders
// — recording everything to the store.
type Session struct {
	Cfg    config.Config
	Client *ibkr.Client
	Store  *store.Store
	Out    io.Writer

	// Advisor is the optional LLM advisory reviewer (internal/agent). nil
	// means "no advisor at all" (no report section, nothing persisted); a
	// non-nil provider — including agent.NullProvider, whose Review reports
	// WHY it's disabled — is consulted once per session, advisory-only. The
	// advisor has NO order path: its output is displayed and persisted, and
	// nothing it returns is ever read by an order-placement code path
	// (CLAUDE.md cardinal rule 2 — risk.Manager.ApproveOrder remains the one
	// gate; see internal/agent's package doc).
	Advisor agent.Provider

	// GetStrategy resolves a registry name to a fresh Strategy instance.
	// Defaults to strategy.Get; tests override it to avoid depending on the
	// global registry (and its side-effect-registered strategy files).
	GetStrategy func(name string) (strategy.Strategy, error)

	// NowFunc returns the current time, defaulting to time.Now. Tests
	// override it for deterministic timestamps in the recorded ledger; the
	// real clock is used in production (determinism of wall-clock time is
	// not required for this package, unlike the backtest engine).
	NowFunc func() time.Time

	// currentAccount is the account the in-flight RunOnce resolved; it scopes
	// ledger walks (position ages, status reconciliation) to that account's
	// rows so a previously used paper account's fills can never corrupt the
	// reconstruction. Session-scoped working state, set once per RunOnce.
	currentAccount string

	// lastRunID is the paper_runs row id the most recent RunOnce actually
	// persisted, or 0 when nothing was persisted (store write failed —
	// RunOnce degrades that to a warning). Callers presenting "this
	// session's run" (the dry-run/execute HTTP handlers) must read this via
	// LastRunID instead of re-deriving it as "the newest row in the table",
	// which can race a concurrent CLI session or silently pick up a PRIOR
	// run's row (and its trace/advisory) after a failed write. (Found by
	// adversarial review 2026-07-09.)
	lastRunID int64
}

// LastRunID returns the paper_runs row id persisted by the most recent
// RunOnce on this Session, or 0 if that run persisted nothing. Only
// meaningful after RunOnce returns.
func (s *Session) LastRunID() int64 { return s.lastRunID }

// ErrNoAccounts is returned when RunOnce needs to auto-discover the paper
// account but the gateway session has none visible.
var ErrNoAccounts = errors.New("paper: no accounts visible to the gateway session")

// ErrAmbiguousAccount is returned when RunOnce needs to auto-discover the
// paper account but the gateway session has more than one visible and
// config.Paper.AccountID was not set to disambiguate.
var ErrAmbiguousAccount = errors.New("paper: account_id is required in config.json's \"paper\" block (multiple accounts visible)")

func (s *Session) now() time.Time {
	if s.NowFunc != nil {
		return s.NowFunc()
	}
	return time.Now()
}

func (s *Session) getStrategy(name string) (strategy.Strategy, error) {
	if s.GetStrategy != nil {
		return s.GetStrategy(name)
	}
	return strategy.Get(name)
}

func (s *Session) out(format string, args ...any) {
	fmt.Fprintf(s.Out, format, args...)
}

// signalOutcome records one signal's journey from strategy intent through
// risk approval to (dry-run or executed) order, for both the human-readable
// report and the persisted ledger.
type signalOutcome struct {
	Strategy     string
	Symbol       string
	TargetWt     float64
	ClampReason  string
	Approved     bool
	RejectReason string
	Order        domain.Order
	RefPrice     float64
	OrderStatus  string // "dry-run" | gateway status | "error: ..."
	OrderID      string
}

// RunOnce executes one paper-trading session end to end. execute selects
// dry-run (default: print intended orders, place nothing) vs. actually
// submitting orders to the gateway.
func (s *Session) RunOnce(ctx context.Context, execute bool) error {
	// Reset per-run working state: a reused Session must not report a prior
	// run's persisted row id.
	s.lastRunID = 0

	// --- 1. MODE GATE ---
	mode, err := s.Cfg.ResolveMode()
	if err != nil {
		return fmt.Errorf("paper: %w", err)
	}
	switch mode {
	case config.ModePaper:
		// ok
	case config.ModeSim:
		return fmt.Errorf(`paper: runtime mode is "sim"; set "mode": "paper" in config.json to run the paper-trading loop`)
	case config.ModeLive:
		return fmt.Errorf("paper: the live order path does not exist yet; Phase 4 is user-gated (see CLAUDE.md cardinal rule 1 and docs/ROADMAP.md)")
	default:
		return fmt.Errorf("paper: unresolved runtime mode")
	}

	if len(s.Cfg.Paper.Strategies) == 0 {
		s.out("No strategies configured under config.json's \"paper\" block — nothing to run (promotion to paper is a human act; see docs/ROADMAP.md gate G1).\n")
		return nil
	}

	// --- 2. GATEWAY: tickle-then-auth ---
	// AuthStatusKeepAlive tickles BEFORE reading auth status, so a session that
	// went dormant since login (observed live: authenticated flips false within
	// ~1-2 min idle) is revived before this fail-closed gate decides. A
	// genuinely dead session stays unauthenticated and still aborts here.
	authStatus, err := s.Client.AuthStatusKeepAlive(ctx)
	if err != nil {
		if errors.Is(err, ibkr.ErrNotAuthenticated) {
			return fmt.Errorf("paper: %w", err)
		}
		return fmt.Errorf("paper: checking gateway auth status: %w", err)
	}
	if !authStatus.Authenticated {
		return fmt.Errorf("%w (gateway reported authenticated=false)", ibkr.ErrNotAuthenticated)
	}

	// --- 3. ACCOUNT ---
	accountID, err := s.resolveAccount(ctx)
	if err != nil {
		return fmt.Errorf("paper: %w", err)
	}
	// currentAccount scopes the ledger walks below (position-age
	// reconstruction, status reconciliation) to this account's rows.
	s.currentAccount = accountID

	// Catch the ledger up with reality BEFORE deriving anything from it: an
	// order placed after hours is recorded as "PreSubmitted"/"Submitted",
	// fills at the next open after that session's process exited, and would
	// otherwise stay non-terminal in the ledger forever — silently disabling
	// the derived position age (and with it every time-stop) on the natural
	// after-close daily schedule. Best-effort: a reconciliation failure
	// warns and continues, it never blocks the session.
	s.reconcileOrderStatuses(ctx)

	// --- 4. BASE CURRENCY + FX RATE ---
	// US symbols price in USD, but the account's ledger is denominated in
	// its BASE currency. The risk manager is a pure single-currency
	// component (weight * equity / price), so the session normalizes
	// everything it hands to ApproveOrder into USD — the trading currency —
	// right here at the boundary. Without a known base currency and (for
	// non-USD accounts) a gateway FX rate, sizing units would be unknown,
	// and mis-sizing is worse than not running: fail closed. (Live,
	// 2026-07-07: BASE(CAD) cash fed to USD prices over-sized the first
	// order by ~1/fx — 280 QQQ ≈ 27% of equity vs the 20% cap.)
	base := s.accountBaseCurrency(ctx, accountID)
	if base == "" {
		return fmt.Errorf("paper: gateway did not report a base currency for account %s — sizing units would be unknown, refusing to run", accountID)
	}
	usdRate := 1.0 // BASE units per 1 USD
	if base != "USD" {
		usdRate, err = s.Client.ExchangeRate(ctx, "USD", base)
		if err != nil {
			return fmt.Errorf("paper: cannot size orders without an FX rate for base currency %s: %w", base, err)
		}
	}

	// --- 5. REAL PORTFOLIO STATE (USD sizing book + BASE ledger figures) ---
	port, cashMoney, equity, skippedNonUSD, err := s.loadPortfolio(ctx, accountID, usdRate)
	if err != nil {
		return fmt.Errorf("paper: loading real portfolio state: %w", err)
	}
	equityUSD := equity / usdRate

	// equityPeak: the risk manager's drawdown kill-switch needs a running
	// peak. A paper session is a fresh process each day with no in-memory
	// history, so the peak is seeded from the highest equity_micros ever
	// persisted (BASE-currency micros, like everything in the stored
	// ledger), or today's equity if this is the very first paper run. The
	// peak is then converted to USD by the SAME day's usdRate as equity
	// itself: dividing both sides of the drawdown comparison by one rate
	// preserves the ratio exactly (equityUSD/peakUSD ==
	// equityBASE/peakBASE), so the kill-switch trips at exactly the same
	// drawdown it would in a BASE-currency check while sizing gets true USD
	// equity.
	equityPeak := equity
	if peakMicros, ok, err := s.Store.LatestPaperEquityPeak(); err != nil {
		s.out("warning: could not read stored equity peak: %v\n", err)
	} else if ok {
		peakFloat := domain.Money(peakMicros).Float()
		if peakFloat > equityPeak {
			equityPeak = peakFloat
		}
	}
	equityPeakUSD := equityPeak / usdRate

	banner := "DRY-RUN (no orders will be placed)"
	runMode := "dry-run"
	if execute {
		banner = fmt.Sprintf("EXECUTING AGAINST PAPER ACCOUNT %s", accountID)
		runMode = "execute"
	}

	s.out("=== TradeForge Paper Session ===\n")
	s.out("Mode:      %s\n", mode)
	s.out("Account:   %s\n", accountID)
	if base == "USD" {
		s.out("Equity:    $%.2f (cash $%.2f)\n", equity, cashMoney.Float())
	} else {
		s.out("Equity:    %.2f %s (base) = %.2f USD @ 1 USD = %.4f %s (gateway rate)\n",
			equity, base, equityUSD, usdRate, base)
	}
	s.out("Execution: %s\n\n", banner)

	// Risk limits come from config.json's optional "risk" block, validated
	// fail-closed (a value looser than internal/risk's hard ceilings refuses
	// the session — never a silent clamp). Printed here so the operator sees
	// the active limits at the decision point.
	riskMgr, err := risk.NewManagerFromConfig(
		s.Cfg.Risk.MaxPositionWeight, s.Cfg.Risk.MaxGrossExposure, s.Cfg.Risk.MaxDrawdown)
	if err != nil {
		return fmt.Errorf("paper: %w", err)
	}
	s.out("Risk:      max position %.0f%% · max gross %.0f%% · kill-switch at -%.0f%% from peak\n\n",
		riskMgr.MaxPositionWeight*100, riskMgr.MaxGrossExposure*100, riskMgr.MaxDrawdown*100)

	var outcomes []signalOutcome
	var warnings []string

	if len(skippedNonUSD) > 0 {
		s.out("note: non-USD positions excluded from the USD sizing book (their value is still counted inside equity): %s\n",
			strings.Join(skippedNonUSD, ", "))
	}

	// traceJSON captures the FIRST traced strategy's deliberation trace this
	// session (agent-committee's CommitteeTrace). One trace column per run:
	// with several traced strategies configured, only the first one that ran
	// and produced a trace is persisted — acceptable for v1, where at most
	// one committee is expected in the paper list at a time.
	var traceJSON string

	for _, stratName := range s.Cfg.Paper.Strategies {
		strat, err := s.getStrategy(stratName)
		if err != nil {
			s.out("strategy %s: %v (skipped)\n", stratName, err)
			continue
		}

		sigs, refPrices, warn, err := s.computeSignals(strat, port)
		if warn != "" {
			warnings = append(warnings, warn)
			s.out("WARNING [%s]: %s\n", stratName, warn)
		}
		if err != nil {
			s.out("strategy %s: %v (skipped)\n", stratName, err)
			continue
		}

		// Capture the deliberation trace of a Traced strategy AFTER its
		// signals were computed (the trace is recorded during
		// OnBar/OnUniverseBar). Best-effort: a trace failure is reported
		// and the session continues.
		if traceJSON == "" {
			if traced, ok := strat.(strategy.Traced); ok {
				if b, err := traced.TraceJSON(); err != nil {
					s.out("warning: could not capture %s's deliberation trace: %v\n", stratName, err)
				} else if len(b) > 0 {
					traceJSON = string(b)
				}
			}
		}

		for _, sig := range sigs {
			refPrice := refPrices[sig.Symbol]
			// All four sizing inputs are USD-denominated: sig's weight is
			// unitless, port is the USD sizing book, refPrice is a US
			// symbol's close, and the peak was converted with today's rate.
			order, approved, reason, clampReason := riskMgr.ApproveOrder(sig, port, refPrice, equityPeakUSD)

			outcome := signalOutcome{
				Strategy:     stratName,
				Symbol:       sig.Symbol,
				TargetWt:     sig.TargetWeight,
				ClampReason:  clampReason,
				Approved:     approved,
				RejectReason: reason,
				RefPrice:     refPrice,
			}

			if !approved {
				outcome.OrderStatus = "rejected"
				outcomes = append(outcomes, outcome)
				continue
			}

			order.Time = s.now().UTC()
			outcome.Order = order
			outcomes = append(outcomes, outcome)
		}
	}

	// --- 8. ORDERS ---
	seq := 0
	dateStamp := s.now().UTC().Format("2006-01-02")
	for i := range outcomes {
		o := &outcomes[i]
		if !o.Approved {
			continue
		}
		seq++

		if !execute {
			o.OrderStatus = "dry-run"
			continue
		}

		conid, err := s.Client.LookupConid(ctx, o.Symbol)
		if err != nil {
			o.OrderStatus = fmt.Sprintf("error: %v", err)
			s.out("order %s %s: conid lookup failed: %v\n", o.Strategy, o.Symbol, err)
			continue
		}

		coid := fmt.Sprintf("tf-%s-%s-%s-%d", dateStamp, o.Strategy, o.Symbol, seq)
		req := ibkr.OrderRequest{
			AccountID: accountID,
			Conid:     conid,
			Side:      o.Order.Side.String(),
			Quantity:  o.Order.Qty,
			OrderType: "MKT",
			TIF:       "DAY",
			COID:      coid,
		}

		result, err := s.Client.PlaceOrder(ctx, req)
		if err != nil {
			o.OrderStatus = fmt.Sprintf("error: %v", err)
			s.out("order %s %s: place order failed: %v\n", o.Strategy, o.Symbol, err)
			continue
		}
		o.OrderID = result.OrderID
		o.OrderStatus = result.Status

		finalStatus, err := s.pollForFillStatus(ctx, result.OrderID, result.Status)
		if err != nil {
			s.out("order %s %s (id %s): poll for fill status: %v\n", o.Strategy, o.Symbol, result.OrderID, err)
		} else {
			o.OrderStatus = finalStatus
		}
	}

	s.printReport(outcomes, execute)

	// --- 8b. AGENT ADVISORY (optional, advisory-only) --- consulted after
	// the plan is fully known. NEVER blocks or fails the session: every
	// failure mode degrades to a disabled Advisory with Err set. Nothing in
	// the Advisory is read by any order code path — it is display + ledger
	// only ("LLMs propose; the risk manager disposes", docs/OPTIONS.md §9).
	advisoryJSON := s.runAdvisor(ctx, mode, base, equity, traceJSON, outcomes, warnings)

	// --- 9. LEDGER --- (BASE-currency figures: equity/cashMoney were never
	// converted; only the sizing book and the peak comparison were USD)
	if err := s.recordLedger(runMode, accountID, equity, cashMoney, outcomes, warnings, traceJSON, advisoryJSON); err != nil {
		s.out("warning: could not persist paper run to the store: %v\n", err)
	}

	placed := 0
	for _, o := range outcomes {
		if o.Approved {
			placed++
		}
	}
	s.out("\nPAPER SESSION COMPLETE — %d order(s) (%s)\n", placed, runMode)

	return nil
}

// runAdvisor consults the optional LLM advisory reviewer (Session.Advisor)
// once, prints the "AGENT ADVISORY" report section, and returns the Advisory
// marshaled as JSON for the ledger ("" when no advisor is wired at all).
// Contract: NEVER blocks beyond advisorTimeout and NEVER fails the session —
// any failure (provider error, marshal error, panic-free degradation inside
// the provider) is reduced to a disabled Advisory with Err set, printed and
// persisted like any other outcome.
func (s *Session) runAdvisor(ctx context.Context, mode config.Mode, baseCurrency string, equity float64, traceJSON string, outcomes []signalOutcome, warnings []string) string {
	if s.Advisor == nil {
		return ""
	}

	in := agent.ReviewInput{
		Mode:               mode.String(),
		AccountEquity:      equity,
		AccountCurrency:    baseCurrency,
		CommitteeTraceJSON: traceJSON,
		DataStaleness:      strings.Join(warnings, "; "),
	}
	for _, o := range outcomes {
		if !o.Approved {
			continue
		}
		in.PlannedOrders = append(in.PlannedOrders, agent.PlannedOrder{
			Symbol:         o.Symbol,
			Side:           o.Order.Side.String(),
			Qty:            o.Order.Qty,
			EstimatedValue: float64(o.Order.Qty) * o.RefPrice,
			Strategy:       o.Strategy,
		})
	}
	in.RecentSessions = s.recentSessionSummaries()

	advCtx, cancel := context.WithTimeout(ctx, advisorTimeout)
	defer cancel()

	advisory, err := s.Advisor.Review(advCtx, in)
	if err != nil {
		// Providers are documented to degrade rather than error; treat an
		// error the same way, belt and suspenders.
		advisory = agent.Advisory{Enabled: false, Err: fmt.Sprintf("advisor error: %v", err)}
	}

	s.out("\nAGENT ADVISORY\n")
	if !advisory.Enabled {
		reason := advisory.Err
		if reason == "" {
			reason = "unknown"
		}
		s.out("advisor disabled: %s\n", reason)
	} else {
		s.out("model:      %s (tokens in/out: %d/%d)\n", advisory.Model, advisory.TokensIn, advisory.TokensOut)
		s.out("summary:    %s\n", advisory.Summary)
		for _, w := range advisory.Warnings {
			s.out("warning:    %s\n", w)
		}
		s.out("confidence: %s\n", advisory.Confidence)
		s.out("(advisory only — no LLM output reaches the broker; risk.Manager and your confirmation remain the gates)\n")
	}

	b, err := json.Marshal(advisory)
	if err != nil {
		s.out("warning: could not marshal advisory for the ledger: %v\n", err)
		return ""
	}
	return string(b)
}

// recentSessionSummaries summarizes the last advisorRecentSessions paper
// runs (oldest first) for the advisor's ReviewInput. Best-effort: any store
// failure returns what was gathered so far — advisory context is never worth
// failing anything over.
func (s *Session) recentSessionSummaries() []agent.RecentSession {
	if s.Store == nil {
		return nil
	}
	runs, err := s.Store.ListPaperRuns(advisorRecentSessions)
	if err != nil {
		return nil
	}
	// ListPaperRuns is newest-first; reverse into chronological order.
	out := make([]agent.RecentSession, 0, len(runs))
	for i := len(runs) - 1; i >= 0; i-- {
		run := runs[i]
		placed, rejected := 0, 0
		orders, err := s.Store.ListPaperOrders(run.Id)
		if err == nil {
			for _, o := range orders {
				if strings.EqualFold(strings.TrimSpace(o.Status), "rejected") {
					rejected++
				} else {
					placed++
				}
			}
		}
		out = append(out, agent.RecentSession{
			Date:           run.Ts.UTC().Format("2006-01-02"),
			OrdersPlaced:   placed,
			OrdersRejected: rejected,
		})
	}
	return out
}

// accountBaseCurrency returns the resolved account's BASE currency (e.g.
// "USD", "CAD") from the gateway's account list, or "" if it cannot be
// determined. RunOnce treats "" as a refusal condition: without a known
// base currency the sizing units are unknown, and the session fails closed
// rather than guess.
func (s *Session) accountBaseCurrency(ctx context.Context, accountID string) string {
	accounts, err := s.Client.Accounts(ctx)
	if err != nil {
		return ""
	}
	for _, a := range accounts {
		if a.ID == accountID {
			return a.Currency
		}
	}
	return ""
}

// resolveAccount returns Cfg.Paper.AccountID if set, otherwise auto-
// discovers it via the gateway's visible accounts (only when exactly one is
// visible).
func (s *Session) resolveAccount(ctx context.Context) (string, error) {
	if s.Cfg.Paper.AccountID != "" {
		return s.Cfg.Paper.AccountID, nil
	}

	accounts, err := s.Client.Accounts(ctx)
	if err != nil {
		return "", fmt.Errorf("discovering account: %w", err)
	}
	switch len(accounts) {
	case 0:
		return "", ErrNoAccounts
	case 1:
		return accounts[0].ID, nil
	default:
		return "", ErrAmbiguousAccount
	}
}

// loadPortfolio builds the USD-denominated sizing portfolio the risk
// manager needs, plus the BASE-currency figures the ledger persists.
// Cardinal rule 6 boundary: the gateway's ledger/positions responses are
// float64 (IBKR's own API shape, mirrored by internal/broker/ibkr), and
// domain.Portfolio itself stays float64 per its own documented Phase-0
// tradeoff — this function does NOT refactor either. The Money conversion
// happens here, at the edge, purely so the caller has a fixed-point cash
// figure to persist to the ledger alongside the float64 Portfolio used for
// signal/risk math.
//
// usdRate is BASE units per 1 USD (1.0 for a USD-base account). Returns:
//   - port: the USD sizing book — only USD-denominated positions, with
//     Cash set so port.Equity(gateway marks) == equityBase/usdRate (see the
//     Cash assignment below)
//   - cashMoney: the account's BASE cash balance, fixed-point, for the
//     ledger
//   - equityBase: whole-account equity in BASE currency, for the ledger,
//     the report header, and the stored kill-switch peak
//   - skippedNonUSD: names of non-USD positions excluded from the sizing
//     book
func (s *Session) loadPortfolio(ctx context.Context, accountID string, usdRate float64) (port *domain.Portfolio, cashMoney domain.Money, equityBase float64, skippedNonUSD []string, err error) {
	// AccountTotals (BASE-first) rather than a bare currency's cash: after
	// the first trade, a non-USD account's USD line goes NEGATIVE (borrowed
	// USD) while the base-currency cash sits untouched — cash alone made
	// equity look like $59 on a $1M account (live, 2026-07-07). netLiq is
	// the whole-account equity figure the kill-switch must see.
	cash, netLiq, err := s.Client.AccountTotals(ctx, accountID)
	if err != nil {
		return nil, 0, 0, nil, fmt.Errorf("account totals: %w", err)
	}
	positions, err := s.Client.Positions(ctx, accountID)
	if err != nil {
		return nil, 0, 0, nil, fmt.Errorf("positions: %w", err)
	}

	// Whole-account BASE equity: the ledger's net liquidation value when
	// the gateway reports one (>0), with cash + the gateway's own position
	// marks as the fallback for gateways/tests that omit it.
	equityBase = netLiq
	if equityBase <= 0 {
		equityBase = cash
		for _, p := range positions {
			if p.Pos == 0 {
				continue
			}
			equityBase += p.MktValue
		}
	}

	// The USD sizing book: only USD-denominated positions enter it, marked
	// at the gateway's own valuation (MktValue/Pos).
	port = domain.NewPortfolio(0)
	usdPositionsValue := 0.0
	for _, p := range positions {
		if p.Pos == 0 {
			continue
		}
		if p.Currency != "USD" {
			// A non-USD position's value is already inside netLiq (and
			// therefore inside the USD cash-equivalent computed below);
			// listing it in a USD book would double-count it in the wrong
			// currency.
			skippedNonUSD = append(skippedNonUSD, fmt.Sprintf("%s (%s)", p.ContractDesc, p.Currency))
			continue
		}
		port.Positions[p.ContractDesc] = domain.Position{
			Symbol:   p.ContractDesc,
			Qty:      int64(p.Pos),
			AvgPrice: p.MktValue / p.Pos,
		}
		usdPositionsValue += p.MktValue
	}

	// Cash here is NOT the account's literal cash balance: it is the
	// definitionally correct USD cash-equivalent of everything in the
	// account that is not a USD position (BASE cash, FX balances, non-USD
	// positions — all already inside netLiq). Defined this way,
	// port.Equity(prices) inside risk.Manager.ApproveOrder reproduces the
	// account's true USD equity (equityBase/usdRate) up to one small,
	// self-consistent ε: ApproveOrder re-marks the traded symbol at its
	// refPrice (the CSV's last close) rather than the gateway's mark, so a
	// held symbol contributes qty·(refPrice − gatewayMark) of drift — it
	// sizes at the same price it marks with, which is the consistent choice.
	// Integer-share assumption: fractional gateway positions (|Pos| < 1)
	// truncate to Qty 0 in the sizing book while their value stays inside
	// Cash via netLiq — a conservative under-mark, acceptable because this
	// platform never creates fractional positions itself.
	port.Cash = equityBase/usdRate - usdPositionsValue

	return port, domain.MoneyFromFloat(cash), equityBase, skippedNonUSD, nil
}

// computeSignals loads stratName's bar data, checks for staleness, and
// computes its signal(s) at the last completed bar exactly once (mirroring
// the backtest engine's step-3 semantics: full history up to and including
// the last completed bar, real current portfolio). Returns the signals, a
// symbol->last-close reference price map, a staleness warning (empty if
// fresh), and an error if data could not be loaded at all.
func (s *Session) computeSignals(strat strategy.Strategy, port *domain.Portfolio) ([]domain.Signal, map[string]float64, string, error) {
	if ms, ok := strat.(strategy.MultiSymbol); ok {
		return s.computeMultiSymbolSignals(ms, port)
	}
	return s.computeSingleSymbolSignals(strat, port)
}

func (s *Session) computeSingleSymbolSignals(strat strategy.Strategy, port *domain.Portfolio) ([]domain.Signal, map[string]float64, string, error) {
	sym, ok := s.Cfg.Paper.Symbols[strat.Name()]
	if !ok || sym == "" {
		return nil, nil, "", fmt.Errorf(
			`single-symbol strategy %q needs an entry in config.json's paper.symbols map (e.g. "symbols": {%q: "SPY"})`,
			strat.Name(), strat.Name())
	}

	bars, warn, err := s.loadSymbolBars(sym)
	if err != nil {
		return nil, nil, warn, err
	}

	warmup := strat.WarmupBars()
	if warmup < 0 {
		warmup = 0
	}
	if len(bars) <= warmup {
		return nil, nil, warn, fmt.Errorf("not enough bars for %s: have %d, need > %d (warmup)", sym, len(bars), warmup)
	}

	last := bars[len(bars)-1]
	ctx := strategy.NewContext(bars, port).
		WithPositionAge(s.positionAges(port, map[string][]domain.Bar{sym: bars}))
	sigs := strat.OnBar(ctx, last)

	refPrices := map[string]float64{sym: last.Close}
	return sigs, refPrices, warn, nil
}

func (s *Session) computeMultiSymbolSignals(ms strategy.MultiSymbol, port *domain.Portfolio) ([]domain.Signal, map[string]float64, string, error) {
	universe := ms.Universe()
	if len(universe) == 0 {
		return nil, nil, "", fmt.Errorf("strategy %q returned an empty Universe()", ms.Name())
	}

	barSets := make(map[string][]domain.Bar, len(universe))
	var warnings []string
	for _, sym := range universe {
		bars, warn, err := s.loadSymbolBars(sym)
		if err != nil {
			return nil, nil, "", fmt.Errorf("loading universe symbol %s: %w", sym, err)
		}
		if warn != "" {
			warnings = append(warnings, warn)
		}
		barSets[sym] = bars
	}

	// The master clock's final tick is the session's "last completed bar"
	// tick, mirroring the engine's step-3 semantics for the multi-symbol
	// path (see internal/backtest.runMulti's SIGNALS step).
	clock := masterClock(barSets)
	if len(clock) == 0 {
		return nil, nil, strings.Join(warnings, "; "), fmt.Errorf("no bars available for universe %v", universe)
	}
	lastTick := clock[len(clock)-1]

	histories := make(map[string][]domain.Bar, len(universe))
	barsAtTick := make(map[string]domain.Bar)
	refPrices := make(map[string]float64, len(universe))
	for _, sym := range universe {
		series := barSets[sym]
		// upperBound = count of bars with Time <= lastTick: this both
		// slices the no-lookahead history (up to and including the current
		// tick, matching the engine's cursor semantics) and identifies
		// whether the symbol has a bar exactly at lastTick (its last
		// element, if any, equals lastTick).
		upperBound := sort.Search(len(series), func(i int) bool { return series[i].Time.After(lastTick) })
		histories[sym] = series[:upperBound]

		if upperBound > 0 && series[upperBound-1].Time.Equal(lastTick) {
			barsAtTick[sym] = series[upperBound-1]
		}
		if upperBound > 0 {
			refPrices[sym] = series[upperBound-1].Close
		}
	}

	ctx := strategy.NewMultiContext(histories, port).
		WithPositionAge(s.positionAges(port, histories))
	sigs := ms.OnUniverseBar(ctx, lastTick, barsAtTick)

	return sigs, refPrices, strings.Join(warnings, "; "), nil
}

// positionAges derives Context.PositionAge for every held symbol from the
// persisted order ledger: walk the ledger newest-first reconstructing the position
// backwards from its current quantity until it reaches flat — the filled BUY
// at that flat point opened the position — then age = number of bars in the
// symbol's loaded series with Time >= that order's UTC date (the fill day's
// bar counts, so the first session after entry sees age 1, matching the
// engine's convention exactly). Every failure mode resolves to age 0 =
// "unknown", the documented safe degradation: strategies then treat the
// position as just-entered and never force a time-stop exit.
func (s *Session) positionAges(port *domain.Portfolio, barsBySym map[string][]domain.Bar) map[string]int {
	ages := make(map[string]int, len(port.Positions))
	if s.Store == nil {
		return ages
	}

	for sym, pos := range port.Positions {
		if pos.Qty == 0 {
			continue
		}
		bars, ok := barsBySym[sym]
		if !ok || len(bars) == 0 {
			continue
		}
		openedAt, found := s.findOpeningFill(sym, pos.Qty)
		if !found {
			continue // age stays 0 = unknown (opened outside TradeForge, or ledger predates DB)
		}
		entryDay := openedAt.UTC().Truncate(24 * time.Hour)
		idx := sort.Search(len(bars), func(i int) bool { return !bars[i].Time.Before(entryDay) })
		ages[sym] = len(bars) - idx
	}
	return ages
}

// findOpeningFill walks the persisted ledger newest-first and reconstructs
// sym's position backwards from currentQty by undoing each FILLED order's
// effect; the filled BUY at the point where the reconstructed prior
// position hits zero is the one that opened the current holding. Returns
// (zero, false) when no flat point is reachable from the ledger.
func (s *Session) findOpeningFill(sym string, currentQty int64) (time.Time, bool) {
	runs, err := s.Store.ListPaperRuns(0) // newest first, unlimited
	if err != nil {
		return time.Time{}, false
	}

	qty := currentQty
	for _, run := range runs {
		if s.currentAccount != "" && run.Account != s.currentAccount {
			continue // another account's history must never enter this reconstruction
		}
		orders, err := s.Store.ListPaperOrders(run.Id)
		if err != nil {
			return time.Time{}, false
		}
		// Orders within a run are stored oldest-first; undo them newest-first.
		for i := len(orders) - 1; i >= 0; i-- {
			o := orders[i]
			if o.Symbol != sym || !isFilledStatus(o.Status) {
				continue
			}
			var before int64
			switch strings.ToUpper(o.Side) {
			case "BUY":
				before = qty - o.Qty
			case "SELL":
				before = qty + o.Qty
			default:
				continue
			}
			if before == 0 && strings.ToUpper(o.Side) == "BUY" {
				return o.Ts, true
			}
			qty = before
		}
	}
	return time.Time{}, false
}

// isFilledStatus reports whether a persisted order status represents a
// COMPLETE fill at the broker. Exact match (case-insensitive), not a
// substring: a partial-fill-style status must not be undone at the order's
// full quantity, which could walk the reconstruction past the true opening
// BUY and overstate the age (the one wrong-positive path — a premature
// time-stop exit). Dry-run, rejected, and error records never moved the
// position and are excluded the same way.
func isFilledStatus(status string) bool {
	return strings.EqualFold(strings.TrimSpace(status), "Filled")
}

// nonTerminalStatus reports whether a persisted order status may still
// change at the broker (submitted/pre-submitted/pending shapes). Terminal
// records (filled, cancelled, rejected, dry-run, error) are never
// re-queried.
func nonTerminalStatus(status string) bool {
	st := strings.ToLower(strings.TrimSpace(status))
	switch {
	case st == "", st == "dry-run", st == "rejected", st == "cancelled", st == "inactive":
		return false
	case st == "filled", strings.HasPrefix(st, "error"):
		return false
	}
	return true
}

// reconcileOrderStatuses catches the persisted ledger up with the gateway:
// every recent order for the current account whose status is still
// non-terminal and which has a broker order id is re-checked against
// LiveOrders and its row updated. Without this, an after-hours session's
// orders freeze at "Submitted" in the ledger even though they filled at the
// next open, and the derived position age silently never activates.
// Best-effort by design: any failure warns and returns.
func (s *Session) reconcileOrderStatuses(ctx context.Context) {
	if s.Store == nil {
		return
	}
	runs, err := s.Store.ListPaperRuns(20) // recent history is enough: older non-terminal rows are already stale beyond DAY-order lifetimes
	if err != nil {
		s.out("warning: could not read ledger for status reconciliation: %v\n", err)
		return
	}

	var pending []store.PaperOrder
	for _, run := range runs {
		if s.currentAccount != "" && run.Account != s.currentAccount {
			continue
		}
		orders, err := s.Store.ListPaperOrders(run.Id)
		if err != nil {
			s.out("warning: could not read orders for run %d: %v\n", run.Id, err)
			return
		}
		for _, o := range orders {
			if o.OrderID != "" && nonTerminalStatus(o.Status) {
				pending = append(pending, o)
			}
		}
	}
	if len(pending) == 0 {
		return
	}

	live, err := s.Client.LiveOrders(ctx)
	if err != nil {
		s.out("warning: could not fetch live orders for status reconciliation: %v\n", err)
		return
	}
	liveByID := make(map[string]ibkr.LiveOrder, len(live))
	for _, lo := range live {
		liveByID[lo.OrderID] = lo
	}

	for _, o := range pending {
		lo, ok := liveByID[o.OrderID]
		if !ok || lo.Status == "" || strings.EqualFold(lo.Status, o.Status) {
			continue
		}
		if err := s.Store.UpdatePaperOrderStatus(o.Id, lo.Status); err != nil {
			s.out("warning: could not update order %s status: %v\n", o.OrderID, err)
			continue
		}
		s.out("reconciled order %s (%s %s): %s -> %s\n", o.OrderID, o.Side, o.Symbol, o.Status, lo.Status)
	}
}

// loadSymbolBars loads sym's CSV from Cfg.Paper.DataDir and checks
// staleness (newest bar older than staleAfter calendar days from now).
func (s *Session) loadSymbolBars(sym string) ([]domain.Bar, string, error) {
	dataDir := s.Cfg.Paper.DataDir
	if dataDir == "" {
		dataDir = "data"
	}
	path := dataDir + "/" + sym + ".csv"

	bars, err := data.LoadCSV(path, sym)
	if err != nil {
		return nil, "", fmt.Errorf("loading %s: %w (run `tradeforge data import` or `data fetch` to populate %s)", sym, err, path)
	}

	warn := ""
	newest := bars[len(bars)-1].Time
	if s.now().UTC().Sub(newest) > staleAfter {
		warn = fmt.Sprintf("stale data for %s — newest bar is %s, run `tradeforge data import`/`data fetch` to refresh", sym, newest.Format("2006-01-02"))
	}

	return bars, warn, nil
}

// pollForFillStatus polls LiveOrders for up to liveOrderPollTimeout,
// returning the most recently observed status for orderID (or initialStatus
// if the order never appears, which is not itself an error — some gateway
// versions omit very-recently-placed orders from a subsequent immediate
// poll).
func (s *Session) pollForFillStatus(ctx context.Context, orderID, initialStatus string) (string, error) {
	deadline := s.now().Add(liveOrderPollTimeout)
	status := initialStatus

	for {
		orders, err := s.Client.LiveOrders(ctx)
		if err != nil {
			return status, err
		}
		for _, o := range orders {
			if o.OrderID == orderID {
				status = o.Status
				if isTerminalStatus(status) {
					return status, nil
				}
			}
		}

		if s.now().After(deadline) {
			return status, nil
		}

		select {
		case <-ctx.Done():
			return status, ctx.Err()
		case <-time.After(liveOrderPollInterval):
		}
	}
}

// isTerminalStatus reports whether an IBKR order status string represents a
// final state that polling should stop on.
func isTerminalStatus(status string) bool {
	switch status {
	case "Filled", "Cancelled", "Rejected", "Inactive":
		return true
	default:
		return false
	}
}

// masterClock returns the sorted union of every bar time across barSets, the
// same concept as backtest.MasterClock but reimplemented locally (rather
// than imported) to avoid this package depending on internal/backtest, which
// CLAUDE.md's task boundary marked read-only/off-limits for this wave.
func masterClock(barSets map[string][]domain.Bar) []time.Time {
	seen := make(map[time.Time]struct{})
	for _, series := range barSets {
		for _, b := range series {
			seen[b.Time] = struct{}{}
		}
	}
	clock := make([]time.Time, 0, len(seen))
	for t := range seen {
		clock = append(clock, t)
	}
	sort.Slice(clock, func(i, j int) bool { return clock[i].Before(clock[j]) })
	return clock
}

// printReport writes the human-readable session report: per-strategy
// signals, risk decisions, and orders.
func (s *Session) printReport(outcomes []signalOutcome, execute bool) {
	if len(outcomes) == 0 {
		s.out("No signals produced this session.\n")
		return
	}

	s.out("Signals & orders:\n")
	s.out("%-16s %-8s %8s %-10s %-8s %6s %10s %-12s %s\n",
		"STRATEGY", "SYMBOL", "TGT WT", "DECISION", "SIDE", "QTY", "REF PRICE", "STATUS", "DETAIL")
	for _, o := range outcomes {
		decision := "REJECTED"
		side := "-"
		qty := int64(0)
		detail := o.RejectReason
		if o.Approved {
			decision = "APPROVED"
			side = o.Order.Side.String()
			qty = o.Order.Qty
			detail = o.ClampReason
		}
		status := o.OrderStatus
		if status == "" {
			status = "-"
		}
		orderIDSuffix := ""
		if o.OrderID != "" {
			orderIDSuffix = " (id " + o.OrderID + ")"
		}
		s.out("%-16s %-8s %7.1f%% %-10s %-8s %6d %10.4f %-12s %s\n",
			o.Strategy, o.Symbol, o.TargetWt*100, decision, side, qty, o.RefPrice, status, detail+orderIDSuffix)
	}
}

// recordLedger persists this session's paper_runs row and one paper_orders
// row per signal outcome (approved or rejected — everything is recorded,
// per the spec, so rejections are visible in `paper status` too). traceJSON
// and advisoryJSON are the agentic layer's per-session artifacts (either may
// be "" — nothing is written to their columns in that case).
func (s *Session) recordLedger(runMode, accountID string, equity float64, cash domain.Money, outcomes []signalOutcome, warnings []string, traceJSON, advisoryJSON string) error {
	detail := struct {
		Warnings []string `json:"warnings,omitempty"`
	}{Warnings: warnings}
	detailJSON, err := json.Marshal(detail)
	if err != nil {
		return fmt.Errorf("marshal run detail: %w", err)
	}

	equityMoney := domain.MoneyFromFloat(equity)
	// Ledger units: BASE currency micros. equity/cash arrive here as the
	// account's BASE-currency figures (loadPortfolio's equityBase/cashMoney),
	// never the USD sizing figures — the persisted equity peak must stay
	// comparable across sessions regardless of the day's FX rate.
	runID, err := s.Store.SavePaperRun(s.now(), runMode, accountID, int64(equityMoney), int64(cash), string(detailJSON))
	if err != nil {
		return fmt.Errorf("save paper run: %w", err)
	}
	s.lastRunID = runID

	for _, o := range outcomes {
		status := o.OrderStatus
		if status == "" {
			status = "rejected"
		}
		side := "-"
		qty := int64(0)
		if o.Approved {
			side = o.Order.Side.String()
			qty = o.Order.Qty
		}
		refPriceMoney := domain.MoneyFromFloat(o.RefPrice)

		orderDetail := o.RejectReason
		if o.ClampReason != "" {
			if orderDetail != "" {
				orderDetail += "; "
			}
			orderDetail += o.ClampReason
		}

		if _, err := s.Store.SavePaperOrder(runID, s.now(), o.Strategy, o.Symbol, side, qty, int64(refPriceMoney), status, o.OrderID, orderDetail); err != nil {
			return fmt.Errorf("save paper order for %s/%s: %w", o.Strategy, o.Symbol, err)
		}
	}

	if traceJSON != "" || advisoryJSON != "" {
		if err := s.Store.SetPaperRunAgent(runID, traceJSON, advisoryJSON); err != nil {
			return fmt.Errorf("save agent artifacts for run %d: %w", runID, err)
		}
	}

	return nil
}
