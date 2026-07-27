package domain

import (
	"fmt"
	"math"
)

// Money is a fixed-point USD amount stored as int64 micros (1e-6 dollars).
// CLAUDE.md cardinal rule 6: broker-facing cash/position accounting (the
// paper/live ledger, order sizing checks recorded to the store) must use
// fixed-point, not float64. Backtest/research code is explicitly exempt —
// Bar, Fill, Position, Portfolio etc. in this package stay float64 (Phase 0
// tradeoff, documented at the top of domain.go) and this type must not be
// threaded into them. Money exists for the boundary where real dollars are
// recorded: the paper session's ledger tables and order-sizing arithmetic.
//
// Range: int64 micros overflows at ~9.2 * 10^12 dollars, far beyond any
// account size this platform will ever hold — MoneyMulQty still guards the
// multiplication explicitly (see its doc) rather than relying on that
// headroom silently.
type Money int64

// microsPerDollar is the fixed-point scale factor: 1 dollar = 1,000,000
// micros.
const microsPerDollar = 1_000_000

// MoneyFromFloat converts a float64 dollar amount to Money, rounding
// half-away-from-zero at the micro (1e-6 dollar) boundary. Half-away-from-
// zero (rather than round-to-even) is chosen so the rounding rule is the
// same one most people expect from "normal" rounding and is symmetric for
// negative amounts (a debit rounds the same way its credit counterpart
// would).
func MoneyFromFloat(f float64) Money {
	scaled := f * microsPerDollar
	if scaled >= 0 {
		return Money(math.Floor(scaled + 0.5))
	}
	return Money(math.Ceil(scaled - 0.5))
}

// Float returns m as a float64 dollar amount. This is a boundary conversion
// back to the float64 world (e.g. to price a domain.Portfolio computation);
// callers performing further money arithmetic should stay in Money instead
// of round-tripping through Float.
func (m Money) Float() float64 {
	return float64(m) / microsPerDollar
}

// String renders m as a fixed 2-decimal-place dollar amount, e.g.
// Money(1234560000).String() == "1234.56". No thousands separators or
// currency symbol — this is a debug/log format, not a display format; the
// dashboard (if it ever renders Money directly) should format from Float()
// with its own presentation rules instead.
func (m Money) String() string {
	neg := m < 0
	v := int64(m)
	if neg {
		v = -v
	}
	whole := v / microsPerDollar
	frac := v % microsPerDollar / (microsPerDollar / 100) // truncate to 2dp
	sign := ""
	if neg {
		sign = "-"
	}
	return fmt.Sprintf("%s%d.%02d", sign, whole, frac)
}

// Add returns m + o.
func (m Money) Add(o Money) Money {
	return m + o
}

// Sub returns m - o.
func (m Money) Sub(o Money) Money {
	return m - o
}

// MoneyMulQty multiplies price (in Money) by an integer share quantity,
// returning an error instead of panicking or silently wrapping on overflow —
// panics are forbidden outside main per CLAUDE.md conventions, and a wrapped
// result would corrupt the ledger silently, which is worse than a loud
// error.
func MoneyMulQty(price Money, qty int64) (Money, error) {
	if price == 0 || qty == 0 {
		return 0, nil
	}
	// MinInt64 * -1 wraps back to MinInt64 AND MinInt64 / -1 wraps too, so
	// the division check below would pass on a silently-wrong result —
	// guard the one pair the check cannot see.
	if (int64(price) == math.MinInt64 && qty == -1) || (qty == math.MinInt64 && int64(price) == -1) {
		return 0, fmt.Errorf("domain: MoneyMulQty overflow: price %s * qty %d overflows int64 micros", price, qty)
	}
	product := int64(price) * qty
	// Overflow check via division: if the multiplication overflowed int64,
	// dividing the (wrapped) product back by qty will not reproduce price.
	if product/qty != int64(price) {
		return 0, fmt.Errorf("domain: MoneyMulQty overflow: price %s * qty %d overflows int64 micros", price, qty)
	}
	return Money(product), nil
}
