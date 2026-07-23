// Package data provides sources of historical price bars: a CSV file loader
// and a deterministic synthetic generator. Later phases add live sources
// (Stooq, IBKR) behind the same BarSource interface.
package data

import (
	"context"

	"tradeforge/internal/domain"
)

// BarSource supplies a complete, chronologically ordered slice of bars for a
// symbol. Phase 0 uses a slice rather than a channel: backtests need random
// access to history and the data sets are small enough to hold in memory.
type BarSource interface {
	Bars(ctx context.Context, symbol string) ([]domain.Bar, error)
}
