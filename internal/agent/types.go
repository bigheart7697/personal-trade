// Package agent implements TradeForge's LLM advisory layer: a provider-
// agnostic reviewer that reads a paper-trading session's planned orders and
// returns a short, structured critique for a human to read before they
// confirm execution.
//
// GUARDRAIL (read this before wiring anything to this package): the advisor
// has NO ORDER PATH. It cannot create, modify, cancel, or approve an order —
// it returns text and a few structured flags that are DISPLAYED to the human
// alongside the dry-run report. The only path from a strategy's intent to a
// broker order is, and remains, risk.Manager.ApproveOrder (CLAUDE.md cardinal
// rule 2). Per docs/OPTIONS.md §9: "LLMs propose; the risk manager disposes."
// v1's guardrail ladder is: strategy intent -> agent-committee's deterministic
// caps -> risk.Manager -> human confirms -> broker. The advisor sits
// alongside that ladder as a read-only reviewer, never inside it.
//
// Cost posture: advisory calls are off by default (config agent.enabled must
// be true AND ANTHROPIC_API_KEY must be set — see FromConfig), capped at
// MaxTokens per call, and made at most once per paper session. See
// docs/AGENT.md for the full design and how to enable it.
package agent

import "context"

// ReviewInput is the compact context handed to a Provider for one advisory
// review: enough for the model to sanity-check a paper session's plan
// without reproducing the whole ledger.
type ReviewInput struct {
	// Mode is the runtime mode the session ran under (always "paper" in
	// practice — internal/paper is mode-gated to paper only).
	Mode string `json:"mode"`

	// AccountEquity and AccountCurrency describe the account the plan would
	// execute against.
	AccountEquity   float64 `json:"accountEquity"`
	AccountCurrency string  `json:"accountCurrency"`

	// PlannedOrders is every signal the session approved (or would approve in
	// dry-run), after risk.Manager sizing. Rejected signals are not included
	// here — they carry no order to review.
	PlannedOrders []PlannedOrder `json:"plannedOrders"`

	// CommitteeTraceJSON is the deliberation trace of a Traced strategy that
	// ran this session (e.g. agent-committee), verbatim JSON, or empty if no
	// such strategy ran.
	CommitteeTraceJSON string `json:"committeeTraceJson,omitempty"`

	// DataStaleness is a human-readable note about how fresh the input data
	// is (empty if fresh). Mirrors internal/paper's own staleness warning.
	DataStaleness string `json:"dataStaleness,omitempty"`

	// RecentSessions summarizes the last few paper sessions (oldest first)
	// so the advisor has some track-record context.
	RecentSessions []RecentSession `json:"recentSessions,omitempty"`
}

// PlannedOrder is one order the risk manager approved (or would approve),
// summarized for the advisor.
type PlannedOrder struct {
	Symbol         string  `json:"symbol"`
	Side           string  `json:"side"`
	Qty            int64   `json:"qty"`
	EstimatedValue float64 `json:"estimatedValue"`
	Strategy       string  `json:"strategy"`
}

// RecentSession summarizes one prior paper session's outcome.
type RecentSession struct {
	Date           string `json:"date"`
	OrdersPlaced   int    `json:"ordersPlaced"`
	OrdersRejected int    `json:"ordersRejected"`
}

// Advisory is a Provider's review result. It is pure information: no field
// here is ever read by an order-placement code path. Enabled reports whether
// a real review actually ran; when false, Err explains why (disabled by
// config, missing API key, or a call failure) and every other field is the
// zero value.
type Advisory struct {
	Enabled    bool     `json:"enabled"`
	Model      string   `json:"model,omitempty"`
	Summary    string   `json:"summary,omitempty"`
	Warnings   []string `json:"warnings,omitempty"`
	Confidence string   `json:"confidence,omitempty"` // "low" | "medium" | "high"
	TokensIn   int      `json:"tokensIn,omitempty"`
	TokensOut  int      `json:"tokensOut,omitempty"`
	// Err is non-empty when the call failed or was skipped. Advisory
	// failures must NEVER block or fail a paper session — callers degrade to
	// Enabled:false with Err set and continue.
	Err string `json:"err,omitempty"`
}

// Provider is the provider-agnostic advisor seam (docs/OPTIONS.md §9): a
// hand-rolled client interface so Anthropic, an OpenAI-compatible endpoint,
// or a local Ollama server are all swappable behind the same call shape.
type Provider interface {
	// Name identifies the provider for logging/status (e.g. "anthropic",
	// "null").
	Name() string
	// Review returns an Advisory for in. It must never panic and must
	// degrade to Advisory{Enabled:false, Err: ...} rather than return an
	// error that could be mistaken for "the review says something is
	// wrong" — the error return exists for truly exceptional caller bugs
	// (e.g. a nil Provider), not for ordinary call failures.
	Review(ctx context.Context, in ReviewInput) (Advisory, error)
}
