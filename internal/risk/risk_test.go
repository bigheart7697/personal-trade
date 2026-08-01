package risk

import (
	"math"
	"testing"

	"tradeforge/internal/domain"
)

func newTestManager() *Manager {
	return &Manager{
		MaxPositionWeight: 0.20,
		MaxGrossExposure:  1.0,
		MaxDrawdown:       0.25,
		DailyLossLimit:    0.03,
	}
}

func TestApproveOrder_WeightClamping(t *testing.T) {
	tests := []struct {
		name         string
		targetWeight float64
		wantApproved bool
		wantSide     domain.Side
		// wantQty is computed from the clamped weight: floor(clamped*equity/price)
		wantQty   int64
		wantClamp bool
	}{
		{
			name:         "weight within cap is not clamped",
			targetWeight: 0.10,
			wantApproved: true,
			wantSide:     domain.Buy,
			wantQty:      100, // floor(0.10*100000/100) = 100
			wantClamp:    false,
		},
		{
			name:         "weight above cap clamps to MaxPositionWeight",
			targetWeight: 0.90,
			wantApproved: true,
			wantSide:     domain.Buy,
			wantQty:      200, // floor(0.20*100000/100) = 200
			wantClamp:    true,
		},
		{
			name:         "weight exactly at cap is not clamped",
			targetWeight: 0.20,
			wantApproved: true,
			wantSide:     domain.Buy,
			wantQty:      200,
			wantClamp:    false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m := newTestManager()
			port := domain.NewPortfolio(100000)
			sig := domain.Signal{Symbol: "T", TargetWeight: tc.targetWeight}

			order, approved, reason, clampReason := m.ApproveOrder(sig, port, 100, 100000)
			if approved != tc.wantApproved {
				t.Fatalf("approved = %v (reason=%q), want %v", approved, reason, tc.wantApproved)
			}
			if (clampReason != "") != tc.wantClamp {
				t.Errorf("clampReason = %q, wantClamp = %v", clampReason, tc.wantClamp)
			}
			if !tc.wantApproved {
				return
			}
			if order.Side != tc.wantSide {
				t.Errorf("side = %v, want %v", order.Side, tc.wantSide)
			}
			if order.Qty != tc.wantQty {
				t.Errorf("qty = %v, want %v", order.Qty, tc.wantQty)
			}
		})
	}
}

func TestApproveOrder_NegativeWeightShortGuard(t *testing.T) {
	t.Run("negative weight while flat produces no order", func(t *testing.T) {
		m := newTestManager()
		port := domain.NewPortfolio(100000)
		sig := domain.Signal{Symbol: "T", TargetWeight: -0.5}

		_, approved, _, clampReason := m.ApproveOrder(sig, port, 100, 100000)
		if approved {
			t.Fatal("expected no order for negative weight while flat, got approved")
		}
		if clampReason == "" {
			t.Error("expected a clamp reason mentioning shorts, got empty string")
		}
	})

	t.Run("negative weight while long 100 shares sells exactly 100", func(t *testing.T) {
		m := newTestManager()
		// Cash 90000 + 100 shares * 100 = 100000 equity.
		port := domain.NewPortfolio(90000)
		port.Positions["T"] = domain.Position{Symbol: "T", Qty: 100, AvgPrice: 100}
		sig := domain.Signal{Symbol: "T", TargetWeight: -0.5}

		order, approved, reason, clampReason := m.ApproveOrder(sig, port, 100, 100000)
		if !approved {
			t.Fatalf("expected flatten order to be approved, got rejection (reason=%q)", reason)
		}
		if clampReason == "" {
			t.Error("expected a clamp reason mentioning shorts, got empty string")
		}
		if order.Side != domain.Sell {
			t.Errorf("side = %v, want Sell", order.Side)
		}
		if order.Qty != 100 {
			t.Errorf("qty = %v, want exactly 100 (flatten, never short)", order.Qty)
		}
	})
}

func TestApproveOrder_ZeroQtyRejected(t *testing.T) {
	m := newTestManager()
	// Cash reflects already having spent 200*100=20000 buying the position,
	// so total equity (cash + position value) is exactly 100000: 80000 cash
	// + 200 shares * 100 = 20000 position value.
	port := domain.NewPortfolio(80000)
	port.Positions["T"] = domain.Position{Symbol: "T", Qty: 200, AvgPrice: 100}

	// Already at the clamped target (0.20 * 100000 / 100 = 200 shares) ->
	// delta is 0 -> should be rejected as a no-op.
	sig := domain.Signal{Symbol: "T", TargetWeight: 0.20}
	_, approved, reason, _ := m.ApproveOrder(sig, port, 100, 100000)
	if approved {
		t.Fatalf("expected zero-qty order to be rejected, got approved (reason=%q)", reason)
	}
}

func TestApproveOrder_GrossExposureRejection(t *testing.T) {
	m := newTestManager()
	m.MaxGrossExposure = 0.5  // tighten the cap so a single position-weight order breaches it
	m.MaxPositionWeight = 0.8 // allow a big enough single position to trip the gross cap

	port := domain.NewPortfolio(100000)
	sig := domain.Signal{Symbol: "T", TargetWeight: 0.80} // 80% gross > 50% cap

	_, approved, reason, _ := m.ApproveOrder(sig, port, 100, 100000)
	if approved {
		t.Fatalf("expected gross exposure rejection, got approved")
	}
	if reason == "" {
		t.Errorf("expected a rejection reason, got empty string")
	}
}

func TestApproveOrder_KillSwitch(t *testing.T) {
	m := newTestManager() // MaxDrawdown = 0.25

	// Equity has fallen to 70% of peak (100000 -> 70000), which is below
	// (1-0.25)*100000 = 75000, so the kill switch should be tripped.
	// Cash (60000) + position value (100 shares * 100 = 10000) = 70000 equity.
	port := domain.NewPortfolio(60000)
	port.Positions["T"] = domain.Position{Symbol: "T", Qty: 100, AvgPrice: 100}
	equityPeak := 100000.0

	t.Run("increasing exposure rejected once tripped", func(t *testing.T) {
		sig := domain.Signal{Symbol: "T", TargetWeight: 0.20} // wants to increase from 100 shares
		_, approved, reason, _ := m.ApproveOrder(sig, port, 100, equityPeak)
		if approved {
			t.Fatalf("expected kill-switch rejection for increasing exposure, got approved")
		}
		if reason == "" {
			t.Errorf("expected a rejection reason, got empty string")
		}
	})

	t.Run("reducing exposure allowed once tripped", func(t *testing.T) {
		sig := domain.Signal{Symbol: "T", TargetWeight: 0.0} // flatten
		order, approved, reason, _ := m.ApproveOrder(sig, port, 100, equityPeak)
		if !approved {
			t.Fatalf("expected reducing/flattening order to be approved even with kill switch tripped, reason=%q", reason)
		}
		if order.Side != domain.Sell {
			t.Errorf("side = %v, want Sell", order.Side)
		}
	})
}

func TestApproveOrder_InvalidPrice(t *testing.T) {
	m := newTestManager()
	port := domain.NewPortfolio(100000)
	sig := domain.Signal{Symbol: "T", TargetWeight: 0.1}

	_, approved, reason, _ := m.ApproveOrder(sig, port, 0, 100000)
	if approved {
		t.Fatalf("expected rejection for zero/invalid price, got approved")
	}
	if reason == "" {
		t.Errorf("expected a rejection reason, got empty string")
	}
}

func TestNewManagerFromConfig(t *testing.T) {
	tests := []struct {
		name           string
		pos, gross, dd float64
		wantErr        bool
		// wantPos/wantGross/wantDD are checked only when wantErr is false.
		wantPos, wantGross, wantDD float64
	}{
		{
			name: "all zero uses defaults",
			pos:  0, gross: 0, dd: 0,
			wantPos: 0.20, wantGross: 1.0, wantDD: 0.25,
		},
		{
			name: "tighter values applied",
			pos:  0.10, gross: 0.80, dd: 0.15,
			wantPos: 0.10, wantGross: 0.80, wantDD: 0.15,
		},
		{
			name:    "position weight at ceiling allowed",
			pos:     0.50,
			wantPos: 0.50, wantGross: 1.0, wantDD: 0.25,
		},
		{
			name: "position weight above ceiling refused",
			pos:  0.51, wantErr: true,
		},
		{
			name: "negative position weight refused",
			pos:  -0.10, wantErr: true,
		},
		{
			name:  "gross exposure above ceiling (leverage) refused",
			gross: 1.10, wantErr: true,
		},
		{
			name: "drawdown above ceiling refused",
			dd:   0.60, wantErr: true,
		},
		{
			name: "drawdown below floor refused",
			dd:   0.01, wantErr: true,
		},
		{
			name:    "drawdown at floor allowed",
			dd:      0.05,
			wantPos: 0.20, wantGross: 1.0, wantDD: 0.05,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := NewManagerFromConfig(tt.pos, tt.gross, tt.dd)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("NewManagerFromConfig(%v, %v, %v): want error, got manager %+v",
						tt.pos, tt.gross, tt.dd, m)
				}
				return
			}
			if err != nil {
				t.Fatalf("NewManagerFromConfig(%v, %v, %v): unexpected error: %v",
					tt.pos, tt.gross, tt.dd, err)
			}
			if m.MaxPositionWeight != tt.wantPos {
				t.Errorf("MaxPositionWeight = %v, want %v", m.MaxPositionWeight, tt.wantPos)
			}
			if m.MaxGrossExposure != tt.wantGross {
				t.Errorf("MaxGrossExposure = %v, want %v", m.MaxGrossExposure, tt.wantGross)
			}
			if m.MaxDrawdown != tt.wantDD {
				t.Errorf("MaxDrawdown = %v, want %v", m.MaxDrawdown, tt.wantDD)
			}
		})
	}
}

func TestNewManagerFromConfig_NaNInfRejected(t *testing.T) {
	nan := math.NaN()
	inf := math.Inf(1)
	cases := []struct {
		name           string
		pos, gross, dd float64
	}{
		{"NaN position weight", nan, 0, 0},
		{"NaN gross exposure", 0, nan, 0},
		{"NaN drawdown", 0, 0, nan},
		{"+Inf position weight", inf, 0, 0},
		{"-Inf drawdown", 0, 0, math.Inf(-1)},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			if m, err := NewManagerFromConfig(tt.pos, tt.gross, tt.dd); err == nil {
				t.Fatalf("NewManagerFromConfig(%v, %v, %v): want error, got manager %+v — a NaN/Inf limit fails OPEN in ApproveOrder",
					tt.pos, tt.gross, tt.dd, m)
			}
		})
	}
}
