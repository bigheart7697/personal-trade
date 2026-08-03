// Package broker defines the seam between the platform and order execution.
// Phase 0 ships SimBroker only; later phases add an IBKR adapter behind the
// same interface (paper first, live only via the CLAUDE.md gate).
package broker

import "tradeforge/internal/domain"

// Broker executes an approved order and reports the resulting fill.
// SubmitAtOpen takes the next bar explicitly (rather than pulling from a
// live feed) so that fill logic — price, slippage, commission — stays
// deterministic and unit-testable in Phase 0.
type Broker interface {
	SubmitAtOpen(order domain.Order, nextBar domain.Bar) (domain.Fill, error)
}
