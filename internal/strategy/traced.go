package strategy

// Traced is an OPTIONAL interface implemented by strategies that maintain a
// structured deliberation trace (e.g. agent-committee's per-rebalance member
// scoring, regime gate, and concentration-cap record). It lets callers like
// internal/paper's session detect and capture trace support without
// importing internal/strategy/strategies concretely (which would otherwise
// pull in every strategy's package-level init() side effects just to check
// one type assertion).
//
// TraceJSON marshals the strategy's most recently recorded trace. It returns
// (nil, nil) — not an error — when no trace has been recorded yet (e.g.
// before the strategy's first rebalance tick).
type Traced interface {
	TraceJSON() ([]byte, error)
}
