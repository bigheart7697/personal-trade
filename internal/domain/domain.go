// Package domain holds the core value types shared across TradeForge: bars,
// orders, fills, positions, portfolios, and strategy signals. These types
// carry no behavior beyond simple accounting — the event-driven pipeline
// (data -> strategy -> risk -> broker -> portfolio -> metrics) is built from
// them but lives in other packages.
//
// Phase 0 tradeoff: prices and cash are float64. This is acceptable for
// research and simulation but must move to fixed-point integers before any
// live capital is at risk (see CLAUDE.md cardinal rule 6).
package domain

import (
	"sort"
	"time"
)

// Bar is one OHLCV price bar for a symbol at a point in time. Time is always
// UTC.
type Bar struct {
	Symbol string
	Time   time.Time
	Open   float64
	High   float64
	Low    float64
	Close  float64
	Volume int64
}

// Side is the direction of an order.
type Side int

const (
	// Buy opens or increases a long position.
	Buy Side = iota
	// Sell closes or decreases a long position.
	Sell
)

// String implements fmt.Stringer.
func (s Side) String() string {
	switch s {
	case Buy:
		return "BUY"
	case Sell:
		return "SELL"
	default:
		return "UNKNOWN"
	}
}

// Order is an approved, sized instruction to trade a quantity of shares.
// Orders are only ever created by risk.Manager.ApproveOrder.
type Order struct {
	ID     string
	Symbol string
	Side   Side
	Qty    int64 // shares, always positive
	Time   time.Time
}

// Fill is the executed result of an Order.
type Fill struct {
	Order      Order
	Price      float64
	Commission float64
	Time       time.Time
}

// Position is the current holding in one symbol.
type Position struct {
	Symbol   string
	Qty      int64
	AvgPrice float64
}

// Signal is strategy intent, not a sized order. TargetWeight is the desired
// fraction of portfolio equity to hold in Symbol, in [-1, 1] (negative values
// are reserved for future short-selling support; Phase 0 strategies emit
// non-negative weights only). risk.Manager is the sole component that turns
// a Signal into an Order.
type Signal struct {
	Symbol       string
	TargetWeight float64
}

// Portfolio tracks cash and open positions. It is the single source of truth
// for account state during a backtest, paper, or live run.
type Portfolio struct {
	Cash      float64
	Positions map[string]Position
}

// NewPortfolio returns a Portfolio seeded with the given starting cash and an
// empty position map.
func NewPortfolio(cash float64) *Portfolio {
	return &Portfolio{
		Cash:      cash,
		Positions: make(map[string]Position),
	}
}

// Equity returns cash plus the mark-to-market value of all positions, priced
// using the supplied last-known prices map (symbol -> price). Symbols held
// but missing from prices are valued at their position's average price as a
// fallback.
//
// Positions are summed in sorted-symbol order: float addition is not
// associative, so summing in Go's randomized map order could mark equity
// last-ULP-differently across identical runs once a portfolio holds several
// positions — a violation of the determinism rule the multi-symbol engine
// depends on.
func (p *Portfolio) Equity(prices map[string]float64) float64 {
	syms := make([]string, 0, len(p.Positions))
	for sym := range p.Positions {
		syms = append(syms, sym)
	}
	sort.Strings(syms)

	total := p.Cash
	for _, sym := range syms {
		pos := p.Positions[sym]
		if pos.Qty == 0 {
			continue
		}
		price, ok := prices[sym]
		if !ok {
			price = pos.AvgPrice
		}
		total += float64(pos.Qty) * price
	}
	return total
}

// ApplyFill updates cash and the relevant position in place to reflect an
// executed Fill. Buys reduce cash by (price*qty + commission) and increase
// the position; sells increase cash by (price*qty - commission) and decrease
// the position. AvgPrice is recomputed on buys using a running weighted
// average; it is left unchanged on sells (and reset when the position is
// fully closed).
func (p *Portfolio) ApplyFill(f Fill) {
	if p.Positions == nil {
		p.Positions = make(map[string]Position)
	}
	sym := f.Order.Symbol
	pos := p.Positions[sym]
	pos.Symbol = sym

	gross := f.Price * float64(f.Order.Qty)

	switch f.Order.Side {
	case Buy:
		p.Cash -= gross + f.Commission
		newQty := pos.Qty + f.Order.Qty
		if newQty != 0 {
			pos.AvgPrice = (pos.AvgPrice*float64(pos.Qty) + gross) / float64(newQty)
		}
		pos.Qty = newQty
	case Sell:
		p.Cash += gross - f.Commission
		pos.Qty -= f.Order.Qty
		if pos.Qty <= 0 {
			pos.Qty = 0
			pos.AvgPrice = 0
		}
	}

	p.Positions[sym] = pos
}
