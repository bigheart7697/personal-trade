package broker

import (
	"fmt"

	"tradeforge/internal/domain"
)

// SimBroker fills orders at the next bar's open price, adjusted for
// slippage (buys pay up, sells receive less), and charges a per-share
// commission with a minimum floor.
//
// Cash sufficiency is checked against AvailableCash, which the caller
// (typically internal/backtest.Engine) must set before each SubmitAtOpen
// call to reflect the portfolio's current cash. SimBroker does not own
// portfolio state — the portfolio remains the single source of truth — but
// the Broker interface's SubmitAtOpen(order, nextBar) signature has no room
// for an extra parameter, so the cash figure to check against is carried as
// broker state, set by the engine immediately before each call.
type SimBroker struct {
	SlippageBps        float64 // basis points, e.g. 5 = 0.05%
	CommissionPerShare float64 // e.g. 0.005
	MinCommission      float64 // e.g. 1.00

	// AvailableCash is the cash the portfolio currently has on hand. The
	// engine sets this before every SubmitAtOpen call. Buy orders whose
	// total cost exceeds it are rejected.
	AvailableCash float64
}

// NewSimBroker returns a SimBroker with the Phase-0 default cost model.
func NewSimBroker(slippageBps float64) *SimBroker {
	return &SimBroker{
		SlippageBps:        slippageBps,
		CommissionPerShare: 0.005,
		MinCommission:      1.0,
	}
}

// Commission computes the commission for an order of qty shares: the
// per-share rate times quantity, floored at MinCommission.
func (b *SimBroker) Commission(qty int64) float64 {
	c := b.CommissionPerShare * float64(qty)
	if c < b.MinCommission {
		return b.MinCommission
	}
	return c
}

// FillPrice computes the executed price for side at nextBar's open, after
// slippage: buys pay up (price rises), sells receive less (price falls).
func (b *SimBroker) FillPrice(side domain.Side, nextOpen float64) float64 {
	slip := b.SlippageBps / 10000.0
	switch side {
	case domain.Buy:
		return nextOpen * (1 + slip)
	case domain.Sell:
		return nextOpen * (1 - slip)
	default:
		return nextOpen
	}
}

// SubmitAtOpen fills order at nextBar's open (adjusted for slippage) and
// charges commission. Buy orders whose total cost (price*qty + commission)
// exceeds b.AvailableCash are rejected with an error so the engine can log
// and skip the fill; sells are never cash-constrained.
func (b *SimBroker) SubmitAtOpen(order domain.Order, nextBar domain.Bar) (domain.Fill, error) {
	if order.Qty <= 0 {
		return domain.Fill{}, fmt.Errorf("broker: order for %s has non-positive qty %d", order.Symbol, order.Qty)
	}

	price := b.FillPrice(order.Side, nextBar.Open)
	commission := b.Commission(order.Qty)

	if order.Side == domain.Buy {
		cost := price*float64(order.Qty) + commission
		if cost > b.AvailableCash {
			return domain.Fill{}, fmt.Errorf(
				"broker: insufficient cash for %s BUY %d @ %.4f (cost %.2f > available %.2f)",
				order.Symbol, order.Qty, price, cost, b.AvailableCash)
		}
	}

	return domain.Fill{
		Order:      order,
		Price:      price,
		Commission: commission,
		Time:       nextBar.Time,
	}, nil
}
