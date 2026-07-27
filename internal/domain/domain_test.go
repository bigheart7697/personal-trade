package domain

import (
	"testing"
	"time"
)

func TestPortfolio_ApplyFill_Buy(t *testing.T) {
	p := NewPortfolio(100000)
	fill := Fill{
		Order:      Order{Symbol: "T", Side: Buy, Qty: 100},
		Price:      50,
		Commission: 5,
		Time:       time.Now().UTC(),
	}

	p.ApplyFill(fill)

	wantCash := 100000.0 - (100*50 + 5)
	if p.Cash != wantCash {
		t.Errorf("Cash = %v, want %v", p.Cash, wantCash)
	}
	pos := p.Positions["T"]
	if pos.Qty != 100 {
		t.Errorf("Qty = %v, want 100", pos.Qty)
	}
	if pos.AvgPrice != 50 {
		t.Errorf("AvgPrice = %v, want 50", pos.AvgPrice)
	}
}

func TestPortfolio_ApplyFill_BuyThenBuy_WeightedAvg(t *testing.T) {
	p := NewPortfolio(100000)
	p.ApplyFill(Fill{Order: Order{Symbol: "T", Side: Buy, Qty: 100}, Price: 50, Commission: 0})
	p.ApplyFill(Fill{Order: Order{Symbol: "T", Side: Buy, Qty: 100}, Price: 60, Commission: 0})

	pos := p.Positions["T"]
	if pos.Qty != 200 {
		t.Fatalf("Qty = %v, want 200", pos.Qty)
	}
	wantAvg := (100.0*50 + 100.0*60) / 200.0
	if pos.AvgPrice != wantAvg {
		t.Errorf("AvgPrice = %v, want %v", pos.AvgPrice, wantAvg)
	}
}

func TestPortfolio_ApplyFill_SellReducesPosition(t *testing.T) {
	p := NewPortfolio(0)
	p.Positions["T"] = Position{Symbol: "T", Qty: 100, AvgPrice: 50}

	p.ApplyFill(Fill{Order: Order{Symbol: "T", Side: Sell, Qty: 40}, Price: 55, Commission: 2})

	wantCash := 40.0*55 - 2
	if p.Cash != wantCash {
		t.Errorf("Cash = %v, want %v", p.Cash, wantCash)
	}
	pos := p.Positions["T"]
	if pos.Qty != 60 {
		t.Errorf("Qty = %v, want 60", pos.Qty)
	}
}

func TestPortfolio_ApplyFill_SellToFlat_ResetsAvgPrice(t *testing.T) {
	p := NewPortfolio(0)
	p.Positions["T"] = Position{Symbol: "T", Qty: 100, AvgPrice: 50}

	p.ApplyFill(Fill{Order: Order{Symbol: "T", Side: Sell, Qty: 100}, Price: 55, Commission: 0})

	pos := p.Positions["T"]
	if pos.Qty != 0 {
		t.Errorf("Qty = %v, want 0", pos.Qty)
	}
	if pos.AvgPrice != 0 {
		t.Errorf("AvgPrice = %v, want 0 after full close", pos.AvgPrice)
	}
}

func TestPortfolio_Equity(t *testing.T) {
	p := NewPortfolio(10000)
	p.Positions["A"] = Position{Symbol: "A", Qty: 100, AvgPrice: 10}
	p.Positions["B"] = Position{Symbol: "B", Qty: 50, AvgPrice: 20}

	prices := map[string]float64{"A": 12, "B": 18}
	got := p.Equity(prices)
	want := 10000.0 + 100*12 + 50*18
	if got != want {
		t.Errorf("Equity() = %v, want %v", got, want)
	}
}

func TestPortfolio_Equity_MissingPriceFallsBackToAvgPrice(t *testing.T) {
	p := NewPortfolio(1000)
	p.Positions["A"] = Position{Symbol: "A", Qty: 10, AvgPrice: 25}

	got := p.Equity(map[string]float64{}) // no price for "A"
	want := 1000.0 + 10*25
	if got != want {
		t.Errorf("Equity() = %v, want %v (should fall back to AvgPrice)", got, want)
	}
}

func TestSide_String(t *testing.T) {
	tests := []struct {
		side Side
		want string
	}{
		{Buy, "BUY"},
		{Sell, "SELL"},
	}
	for _, tc := range tests {
		if got := tc.side.String(); got != tc.want {
			t.Errorf("Side(%d).String() = %q, want %q", tc.side, got, tc.want)
		}
	}
}
