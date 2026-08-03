package broker

import (
	"testing"
	"time"

	"tradeforge/internal/domain"
)

func mkNextBar(open float64) domain.Bar {
	return domain.Bar{
		Symbol: "T",
		Time:   time.Date(2020, 1, 2, 0, 0, 0, 0, time.UTC),
		Open:   open,
		High:   open + 1,
		Low:    open - 1,
		Close:  open,
		Volume: 1000,
	}
}

func TestSubmitAtOpen_CommissionMinimum(t *testing.T) {
	tests := []struct {
		name           string
		qty            int64
		perShare       float64
		minCommission  float64
		wantCommission float64
	}{
		{
			name:           "below minimum floors to MinCommission",
			qty:            10,
			perShare:       0.005,
			minCommission:  1.0,
			wantCommission: 1.0, // 10*0.005 = 0.05 < 1.00 floor
		},
		{
			name:           "above minimum uses per-share rate",
			qty:            1000,
			perShare:       0.005,
			minCommission:  1.0,
			wantCommission: 5.0, // 1000*0.005 = 5.00 > 1.00 floor
		},
		{
			name:           "exactly at minimum",
			qty:            200,
			perShare:       0.005,
			minCommission:  1.0,
			wantCommission: 1.0, // 200*0.005 = 1.00 == floor
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			b := &SimBroker{
				SlippageBps:        0,
				CommissionPerShare: tc.perShare,
				MinCommission:      tc.minCommission,
				AvailableCash:      1_000_000,
			}
			order := domain.Order{Symbol: "T", Side: domain.Buy, Qty: tc.qty}
			fill, err := b.SubmitAtOpen(order, mkNextBar(100))
			if err != nil {
				t.Fatalf("SubmitAtOpen() error = %v", err)
			}
			if fill.Commission != tc.wantCommission {
				t.Errorf("Commission = %v, want %v", fill.Commission, tc.wantCommission)
			}
		})
	}
}

func TestSubmitAtOpen_InsufficientCashRejected(t *testing.T) {
	b := &SimBroker{
		SlippageBps:        0,
		CommissionPerShare: 0.005,
		MinCommission:      1.0,
		AvailableCash:      500, // not enough for 100 shares @ 100 = 10000
	}
	order := domain.Order{Symbol: "T", Side: domain.Buy, Qty: 100}

	_, err := b.SubmitAtOpen(order, mkNextBar(100))
	if err == nil {
		t.Fatal("expected an error for insufficient cash, got nil")
	}
}

func TestSubmitAtOpen_SufficientCashApproved(t *testing.T) {
	b := &SimBroker{
		SlippageBps:        0,
		CommissionPerShare: 0.005,
		MinCommission:      1.0,
		AvailableCash:      100000,
	}
	order := domain.Order{Symbol: "T", Side: domain.Buy, Qty: 100}

	fill, err := b.SubmitAtOpen(order, mkNextBar(100))
	if err != nil {
		t.Fatalf("SubmitAtOpen() error = %v, want nil", err)
	}
	if fill.Price != 100 {
		t.Errorf("Price = %v, want 100 (no slippage configured)", fill.Price)
	}
}

func TestSubmitAtOpen_SellNeverCashConstrained(t *testing.T) {
	b := &SimBroker{
		SlippageBps:        0,
		CommissionPerShare: 0.005,
		MinCommission:      1.0,
		AvailableCash:      0, // no cash at all
	}
	order := domain.Order{Symbol: "T", Side: domain.Sell, Qty: 100}

	_, err := b.SubmitAtOpen(order, mkNextBar(100))
	if err != nil {
		t.Fatalf("SubmitAtOpen() sell error = %v, want nil (sells are not cash-constrained)", err)
	}
}

func TestSubmitAtOpen_ZeroQtyRejected(t *testing.T) {
	b := NewSimBroker(5)
	b.AvailableCash = 100000
	order := domain.Order{Symbol: "T", Side: domain.Buy, Qty: 0}

	_, err := b.SubmitAtOpen(order, mkNextBar(100))
	if err == nil {
		t.Fatal("expected an error for zero qty, got nil")
	}
}

func TestFillPrice_SlippageDirection(t *testing.T) {
	b := &SimBroker{SlippageBps: 100} // 1%

	buyPrice := b.FillPrice(domain.Buy, 100)
	wantBuy := 101.0
	if !almostEqual(buyPrice, wantBuy, 1e-9) {
		t.Errorf("buy fill price = %v, want %v (buys pay up)", buyPrice, wantBuy)
	}

	sellPrice := b.FillPrice(domain.Sell, 100)
	wantSell := 99.0
	if !almostEqual(sellPrice, wantSell, 1e-9) {
		t.Errorf("sell fill price = %v, want %v (sells receive less)", sellPrice, wantSell)
	}
}

func almostEqual(a, b, tol float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= tol
}
