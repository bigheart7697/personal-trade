package strategy

import (
	"testing"
	"time"

	"tradeforge/internal/domain"
)

func ctxBar(symbol string, day int) domain.Bar {
	return domain.Bar{
		Symbol: symbol,
		Time:   time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC).AddDate(0, 0, day),
		Open:   100, High: 102, Low: 98, Close: 101,
		Volume: 1000,
	}
}

func TestContext_PositionAge(t *testing.T) {
	history := []domain.Bar{ctxBar("T", 0), ctxBar("T", 1)}
	port := domain.NewPortfolio(100000)

	t.Run("defaults to 0 without WithPositionAge", func(t *testing.T) {
		ctx := NewContext(history, port)
		if got := ctx.PositionAge("T"); got != 0 {
			t.Errorf("PositionAge(T) = %d, want 0 (unknown by default)", got)
		}
	})

	t.Run("WithPositionAge round-trips and chains", func(t *testing.T) {
		ctx := NewContext(history, port).WithPositionAge(map[string]int{"T": 3, "X": 7})
		if ctx == nil {
			t.Fatal("WithPositionAge returned nil, want the receiver (chainable)")
		}
		if got := ctx.PositionAge("T"); got != 3 {
			t.Errorf("PositionAge(T) = %d, want 3", got)
		}
		if got := ctx.PositionAge("X"); got != 7 {
			t.Errorf("PositionAge(X) = %d, want 7", got)
		}
		if got := ctx.PositionAge("UNKNOWN"); got != 0 {
			t.Errorf("PositionAge(UNKNOWN) = %d, want 0 (symbol not in map)", got)
		}
	})

	t.Run("nil map is safe", func(t *testing.T) {
		ctx := NewContext(history, port).WithPositionAge(nil)
		if got := ctx.PositionAge("T"); got != 0 {
			t.Errorf("PositionAge(T) = %d, want 0 (nil ages map)", got)
		}
	})

	t.Run("works on a multi-symbol Context", func(t *testing.T) {
		histories := map[string][]domain.Bar{
			"A": {ctxBar("A", 0)},
			"B": {ctxBar("B", 0)},
		}
		ctx := NewMultiContext(histories, port).WithPositionAge(map[string]int{"A": 2})
		if got := ctx.PositionAge("A"); got != 2 {
			t.Errorf("PositionAge(A) = %d, want 2", got)
		}
		if got := ctx.PositionAge("B"); got != 0 {
			t.Errorf("PositionAge(B) = %d, want 0 (not held)", got)
		}
	})
}
