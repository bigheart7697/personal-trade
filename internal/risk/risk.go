// Package risk contains the sole path by which a strategy's intent (a
// Signal) becomes an order the broker can execute. Every order in
// TradeForge — backtest, paper, or live — must be produced by
// Manager.ApproveOrder. See CLAUDE.md cardinal rule 2.
package risk

import (
	"fmt"
	"math"

	"tradeforge/internal/domain"
)

// Manager sizes and approves (or rejects) trading intent.
type Manager struct {
	// MaxPositionWeight caps the absolute target weight of any single
	// position as a fraction of equity (e.g. 0.20 = 20%).
	MaxPositionWeight float64
	// MaxGrossExposure caps total gross exposure (sum of |position value|)
	// as a fraction of equity (e.g. 1.0 = fully invested, no leverage).
	MaxGrossExposure float64
	// MaxDrawdown is the kill-switch threshold: once equity falls below
	// (1-MaxDrawdown)*equityPeak, only orders that reduce or flatten
	// exposure are approved (e.g. 0.25 = a 25% drawdown trips the switch).
	MaxDrawdown float64
	// DailyLossLimit is a soft stop reserved for future intraday use; not
	// enforced in Phase 0 (no intraday data), but recorded here as
	// documented config per the design.
	DailyLossLimit float64
}

// NewManager returns a Manager with the Phase-0 default limits.
func NewManager() *Manager {
	return &Manager{
		MaxPositionWeight: 0.20,
		MaxGrossExposure:  1.0,
		MaxDrawdown:       0.25,
		DailyLossLimit:    0.03,
	}
}

// Hard ceilings for user-configurable limits (NewManagerFromConfig). A
// config value looser than these is refused, never clamped: the platform is
// long-only with no leverage, and a kill-switch that only trips beyond a 50%
// drawdown is not a kill-switch.
const (
	configMaxPositionWeightCeiling = 0.50
	configMaxGrossExposureCeiling  = 1.00
	configMaxDrawdownCeiling       = 0.50
	configMaxDrawdownFloor         = 0.05
)

// NewManagerFromConfig returns a Manager whose limits come from user config
// (config.json's "risk" block). The contract is FAIL-CLOSED: a zero value
// means "use the built-in default" (see NewManager), while any non-zero
// value must be at least as tight as the hard ceilings above — a looser or
// nonsensical value is an error that refuses the session, never a silent
// clamp to something the user did not write. Config can therefore only ever
// tighten risk relative to the ceilings, and omitting the block entirely
// keeps today's behavior bit-for-bit.
func NewManagerFromConfig(posWeight, grossExposure, drawdown float64) (*Manager, error) {
	m := NewManager()

	// NaN/Inf are rejected up front: NaN passes a `!= 0` check, fails BOTH
	// range comparisons below (all comparisons with NaN are false), and a
	// NaN limit then disables every ApproveOrder comparison it appears in —
	// fail-OPEN, the exact opposite of this function's contract. JSON can't
	// encode NaN today, but this is the exported perimeter validator and
	// must not depend on its callers' parsers. (Found by adversarial review
	// 2026-07-09.)
	for name, v := range map[string]float64{
		"max_position_weight": posWeight,
		"max_gross_exposure":  grossExposure,
		"max_drawdown":        drawdown,
	} {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("risk: config %s must be a finite number, got %v", name, v)
		}
	}

	if posWeight != 0 {
		if posWeight < 0 || posWeight > configMaxPositionWeightCeiling {
			return nil, fmt.Errorf(
				"risk: config max_position_weight %.4f out of range (0, %.2f]", posWeight, configMaxPositionWeightCeiling)
		}
		m.MaxPositionWeight = posWeight
	}
	if grossExposure != 0 {
		if grossExposure < 0 || grossExposure > configMaxGrossExposureCeiling {
			return nil, fmt.Errorf(
				"risk: config max_gross_exposure %.4f out of range (0, %.2f]", grossExposure, configMaxGrossExposureCeiling)
		}
		m.MaxGrossExposure = grossExposure
	}
	if drawdown != 0 {
		if drawdown < configMaxDrawdownFloor || drawdown > configMaxDrawdownCeiling {
			return nil, fmt.Errorf(
				"risk: config max_drawdown %.4f out of range [%.2f, %.2f]", drawdown, configMaxDrawdownFloor, configMaxDrawdownCeiling)
		}
		m.MaxDrawdown = drawdown
	}
	return m, nil
}

// ApproveOrder converts a strategy's Signal into a sized domain.Order, or
// rejects it with a human-readable reason. price is the reference price
// used for sizing (Phase 0: the signal bar's close). equityPeak is the
// highest equity mark observed so far in this run, used for the drawdown
// kill switch.
//
// Returns (order, approved, reason, clampReason). reason is non-empty only
// when the signal was rejected (approved == false). clampReason is
// non-empty whenever the requested target weight had to be altered before
// sizing — either capped at MaxPositionWeight or zeroed because shorts are
// not yet supported — so callers can surface clamping instead of it
// happening silently. A clamp and an approval are not mutually exclusive:
// a clamped weight usually still produces an approved (smaller) order.
//
// Sizing: negative target weights are clamped to 0 (Phase 0 has no short
// support — a negative weight is treated as "flatten"), positive weights
// are capped at MaxPositionWeight, then converted to a desired share count
// via floor(clampedWeight * equity / price); the order quantity is the
// difference between that and the current position. A zero-quantity result
// (already at target) is rejected as a no-op.
//
// Once the kill switch has tripped (current equity < (1-MaxDrawdown) *
// equityPeak), only orders that reduce or flatten the existing position are
// approved; anything that would increase exposure is rejected.
//
// Orders that would push gross exposure (post-trade) above MaxGrossExposure
// are rejected.
func (m *Manager) ApproveOrder(sig domain.Signal, port *domain.Portfolio, price float64, equityPeak float64) (domain.Order, bool, string, string) {
	var zero domain.Order

	if price <= 0 {
		return zero, false, fmt.Sprintf("invalid reference price %.4f for %s", price, sig.Symbol), ""
	}

	prices := map[string]float64{sig.Symbol: price}
	equity := port.Equity(prices)
	if equity <= 0 {
		return zero, false, fmt.Sprintf("non-positive equity %.2f, cannot size order", equity), ""
	}

	clampedWeight := sig.TargetWeight
	clampReason := ""
	switch {
	case clampedWeight < 0:
		clampedWeight = 0
		clampReason = fmt.Sprintf(
			"shorts not yet supported: target weight %.2f clamped to 0 (flatten)", sig.TargetWeight)
	case clampedWeight > m.MaxPositionWeight:
		clampedWeight = m.MaxPositionWeight
		clampReason = fmt.Sprintf(
			"target weight %.2f clamped to max position weight %.2f", sig.TargetWeight, m.MaxPositionWeight)
	}

	currentQty := int64(0)
	if pos, ok := port.Positions[sig.Symbol]; ok {
		currentQty = pos.Qty
	}

	desiredQty := int64(math.Floor(clampedWeight * equity / price))
	deltaQty := desiredQty - currentQty

	if deltaQty == 0 {
		return zero, false, "no change: already at target position", clampReason
	}

	side := domain.Buy
	qty := deltaQty
	if deltaQty < 0 {
		side = domain.Sell
		qty = -deltaQty
	}

	// Kill switch: once equity has fallen far enough from its peak, only
	// orders that reduce or flatten existing exposure are allowed.
	killSwitchTripped := equityPeak > 0 && equity < (1-m.MaxDrawdown)*equityPeak
	if killSwitchTripped {
		increasesExposure := isIncreasingExposure(currentQty, deltaQty)
		if increasesExposure {
			return zero, false, fmt.Sprintf(
				"kill switch tripped (equity %.2f < %.2f%% of peak %.2f): only reducing orders allowed",
				equity, (1-m.MaxDrawdown)*100, equityPeak), clampReason
		}
	}

	// Gross exposure check: sum |position value| across the book after this
	// trade, as a fraction of equity.
	grossAfter := grossExposureAfter(port, prices, sig.Symbol, currentQty+deltaQty, price)
	grossPct := grossAfter / equity
	if grossPct > m.MaxGrossExposure+1e-9 {
		return zero, false, fmt.Sprintf(
			"gross exposure %.2f%% would exceed cap %.2f%%", grossPct*100, m.MaxGrossExposure*100), clampReason
	}

	order := domain.Order{
		Symbol: sig.Symbol,
		Side:   side,
		Qty:    qty,
	}
	return order, true, "", clampReason
}

// isIncreasingExposure reports whether applying deltaQty to a position of
// currentQty shares moves the position further from zero (i.e. adds risk)
// rather than toward zero (reducing/flattening).
func isIncreasingExposure(currentQty, deltaQty int64) bool {
	newQty := currentQty + deltaQty
	return abs64(newQty) > abs64(currentQty)
}

// grossExposureAfter computes total gross exposure (sum of |qty*price|
// across all positions) as if symbol's position were replaced by newQty
// shares at price, leaving all other positions priced at their last-known
// (or average) price.
func grossExposureAfter(port *domain.Portfolio, prices map[string]float64, symbol string, newQty int64, price float64) float64 {
	var gross float64
	seenSymbol := false
	for sym, pos := range port.Positions {
		if sym == symbol {
			seenSymbol = true
			gross += math.Abs(float64(newQty)) * price
			continue
		}
		p, ok := prices[sym]
		if !ok {
			p = pos.AvgPrice
		}
		gross += math.Abs(float64(pos.Qty)) * p
	}
	if !seenSymbol {
		gross += math.Abs(float64(newQty)) * price
	}
	return gross
}

func abs64(v int64) int64 {
	if v < 0 {
		return -v
	}
	return v
}
