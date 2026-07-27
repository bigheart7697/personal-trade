package domain

import (
	"math"
	"strings"
	"testing"
)

func TestMoneyFromFloat_RoundingEdges(t *testing.T) {
	tests := []struct {
		name  string
		input float64
		want  Money
	}{
		{"exact whole dollar", 100.0, Money(100_000_000)},
		{"exact 2dp", 1234.56, Money(1_234_560_000)},
		{"rounds half up at micro boundary", 1.0000005, Money(1_000_001)},
		{"rounds half up at micro boundary (down side)", 1.0000004, Money(1_000_000)},
		{"negative rounds half away from zero", -1.0000005, Money(-1_000_001)},
		{"negative truncation direction", -1.0000004, Money(-1_000_000)},
		{"zero", 0.0, Money(0)},
		{"negative whole dollar", -50.0, Money(-50_000_000)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := MoneyFromFloat(tt.input)
			if got != tt.want {
				t.Errorf("MoneyFromFloat(%v) = %d, want %d", tt.input, got, tt.want)
			}
		})
	}
}

func TestMoney_FloatRoundTrip(t *testing.T) {
	m := MoneyFromFloat(1234.56)
	if got := m.Float(); math.Abs(got-1234.56) > 1e-9 {
		t.Errorf("Float() = %v, want ~1234.56", got)
	}
}

func TestMoney_String(t *testing.T) {
	tests := []struct {
		name string
		m    Money
		want string
	}{
		{"positive with cents", Money(1_234_560_000), "1234.56"},
		{"zero", Money(0), "0.00"},
		{"negative with cents", Money(-1_234_560_000), "-1234.56"},
		{"truncates beyond 2dp", Money(1_000_001), "1.00"},
		{"whole dollar", Money(100_000_000), "100.00"},
		{"small negative", Money(-500_000), "-0.50"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.m.String(); got != tt.want {
				t.Errorf("String() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestMoney_AddSub(t *testing.T) {
	a := MoneyFromFloat(100.50)
	b := MoneyFromFloat(50.25)

	if got, want := a.Add(b), MoneyFromFloat(150.75); got != want {
		t.Errorf("Add() = %v, want %v", got, want)
	}
	if got, want := a.Sub(b), MoneyFromFloat(50.25); got != want {
		t.Errorf("Sub() = %v, want %v", got, want)
	}
	if got, want := b.Sub(a), MoneyFromFloat(-50.25); got != want {
		t.Errorf("Sub() (negative) = %v, want %v", got, want)
	}
}

func TestMoneyMulQty(t *testing.T) {
	tests := []struct {
		name    string
		price   Money
		qty     int64
		want    Money
		wantErr bool
	}{
		{"simple", MoneyFromFloat(10.00), 5, MoneyFromFloat(50.00), false},
		{"zero price", Money(0), 100, Money(0), false},
		{"zero qty", MoneyFromFloat(10.00), 0, Money(0), false},
		{"negative price (sell proceeds sign convention)", MoneyFromFloat(-10.00), 5, MoneyFromFloat(-50.00), false},
		{"negative qty", MoneyFromFloat(10.00), -5, MoneyFromFloat(-50.00), false},
		{"overflow", Money(math.MaxInt64 / 2), 3, 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := MoneyMulQty(tt.price, tt.qty)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("MoneyMulQty() error = nil, want overflow error")
				}
				if !strings.Contains(err.Error(), "overflow") {
					t.Errorf("MoneyMulQty() error = %q, want it to mention overflow", err.Error())
				}
				return
			}
			if err != nil {
				t.Fatalf("MoneyMulQty() error = %v, want nil", err)
			}
			if got != tt.want {
				t.Errorf("MoneyMulQty(%v, %d) = %v, want %v", tt.price, tt.qty, got, tt.want)
			}
		})
	}
}

func TestMoneyMulQty_OverflowDoesNotPanic(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("MoneyMulQty panicked: %v", r)
		}
	}()
	_, err := MoneyMulQty(Money(math.MaxInt64), math.MaxInt64)
	if err == nil {
		t.Fatal("MoneyMulQty() error = nil, want overflow error")
	}
}
