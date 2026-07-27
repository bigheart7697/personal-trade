package strategy

import (
	"math"
	"testing"
	"time"

	"tradeforge/internal/domain"
)

func almostEqual(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

func TestSMA(t *testing.T) {
	tests := []struct {
		name   string
		closes []float64
		n      int
		want   float64
		wantOk bool
	}{
		{
			name:   "hand computed n=3",
			closes: []float64{10, 20, 30, 40},
			n:      3,
			// last 3: 20,30,40 -> avg 30
			want:   30,
			wantOk: true,
		},
		{
			name:   "n equals length",
			closes: []float64{2, 4, 6},
			n:      3,
			want:   4,
			wantOk: true,
		},
		{
			name:   "short input",
			closes: []float64{1, 2},
			n:      3,
			wantOk: false,
		},
		{
			name:   "n<=0 guard",
			closes: []float64{1, 2, 3},
			n:      0,
			wantOk: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := SMA(tc.closes, tc.n)
			if ok != tc.wantOk {
				t.Fatalf("SMA() ok = %v, want %v", ok, tc.wantOk)
			}
			if ok && !almostEqual(got, tc.want, 1e-9) {
				t.Errorf("SMA() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestEMA(t *testing.T) {
	// Hand-compute EMA(n=3) for closes = [1,2,3,4,5].
	// seed = SMA(first 3) = (1+2+3)/3 = 2
	// k = 2/(3+1) = 0.5
	// step for 4: 4*0.5 + 2*0.5 = 3
	// step for 5: 5*0.5 + 3*0.5 = 4
	closes := []float64{1, 2, 3, 4, 5}
	got, ok := EMA(closes, 3)
	if !ok {
		t.Fatalf("EMA() ok = false, want true")
	}
	want := 4.0
	if !almostEqual(got, want, 1e-9) {
		t.Errorf("EMA() = %v, want %v", got, want)
	}

	if _, ok := EMA([]float64{1, 2}, 3); ok {
		t.Errorf("EMA() with short input: ok = true, want false")
	}
	if _, ok := EMA(closes, 0); ok {
		t.Errorf("EMA() with n<=0: ok = true, want false")
	}
}

func TestRSI(t *testing.T) {
	// Hand-compute simple-average RSI (Cutler's: plain arithmetic means of
	// gains/losses, the Connors RSI(2) convention — not Wilder smoothing)
	// with n=2 over closes = [100, 102, 101].
	// changes: 102-100=+2, 101-102=-1
	// avgGain = 2/2 = 1, avgLoss = 1/2 = 0.5
	// RS = 1/0.5 = 2, RSI = 100 - 100/(1+2) = 100 - 33.333... = 66.666...
	closes := []float64{100, 102, 101}
	got, ok := RSI(closes, 2)
	if !ok {
		t.Fatalf("RSI() ok = false, want true")
	}
	want := 66.6666666667
	if !almostEqual(got, want, 1e-6) {
		t.Errorf("RSI() = %v, want %v", got, want)
	}

	// All gains -> RSI = 100.
	allUp := []float64{10, 11, 12, 13}
	got, ok = RSI(allUp, 3)
	if !ok {
		t.Fatalf("RSI() ok = false, want true")
	}
	if !almostEqual(got, 100, 1e-9) {
		t.Errorf("RSI() all-gains = %v, want 100", got)
	}

	// Flat series -> RSI = 50 (neutral).
	flat := []float64{5, 5, 5, 5}
	got, ok = RSI(flat, 3)
	if !ok {
		t.Fatalf("RSI() ok = false, want true")
	}
	if !almostEqual(got, 50, 1e-9) {
		t.Errorf("RSI() flat = %v, want 50", got)
	}

	// Short input guard: need n+1 values.
	if _, ok := RSI([]float64{1, 2}, 3); ok {
		t.Errorf("RSI() with short input: ok = true, want false")
	}
	if _, ok := RSI(closes, 0); ok {
		t.Errorf("RSI() with n<=0: ok = true, want false")
	}
}

func mkBar(day int, o, h, l, c float64) domain.Bar {
	return domain.Bar{
		Symbol: "T",
		Time:   time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC).AddDate(0, 0, day),
		Open:   o, High: h, Low: l, Close: c,
		Volume: 1000,
	}
}

func TestATR(t *testing.T) {
	// Hand-computed: 3 bars, n=2 (needs n+1=3 bars).
	// bar0: close 10
	// bar1: H=12,L=9,close=11 -> TR = max(H-L=3, |H-prevClose|=2, |L-prevClose|=1) = 3
	// bar2: H=13,L=10,close=12 -> TR = max(H-L=3, |H-prevClose(11)|=2, |L-prevClose(11)|=1) = 3
	// ATR = (3+3)/2 = 3
	bars := []domain.Bar{
		mkBar(0, 10, 10, 10, 10),
		mkBar(1, 11, 12, 9, 11),
		mkBar(2, 12, 13, 10, 12),
	}
	got, ok := ATR(bars, 2)
	if !ok {
		t.Fatalf("ATR() ok = false, want true")
	}
	want := 3.0
	if !almostEqual(got, want, 1e-9) {
		t.Errorf("ATR() = %v, want %v", got, want)
	}

	if _, ok := ATR(bars[:1], 2); ok {
		t.Errorf("ATR() with short input: ok = true, want false")
	}
	if _, ok := ATR(bars, 0); ok {
		t.Errorf("ATR() with n<=0: ok = true, want false")
	}
}

func TestStdDev(t *testing.T) {
	// Population stdev of [2,4,4,4,5,5,7,9] = 2 (classic example).
	xs := []float64{2, 4, 4, 4, 5, 5, 7, 9}
	got, ok := StdDev(xs, len(xs))
	if !ok {
		t.Fatalf("StdDev() ok = false, want true")
	}
	want := 2.0
	if !almostEqual(got, want, 1e-9) {
		t.Errorf("StdDev() = %v, want %v", got, want)
	}

	if _, ok := StdDev([]float64{1}, 1); ok {
		t.Errorf("StdDev() with n<=1: ok = true, want false")
	}
	if _, ok := StdDev(xs, 100); ok {
		t.Errorf("StdDev() with short input: ok = true, want false")
	}
}

func TestHighs(t *testing.T) {
	tests := []struct {
		name string
		bars []domain.Bar
		want []float64
	}{
		{
			name: "empty",
			bars: nil,
			want: []float64{},
		},
		{
			name: "several bars",
			bars: []domain.Bar{
				mkBar(0, 10, 12, 9, 11),
				mkBar(1, 11, 13, 10, 12),
			},
			want: []float64{12, 13},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := Highs(tc.bars)
			if len(got) != len(tc.want) {
				t.Fatalf("Highs() len = %d, want %d", len(got), len(tc.want))
			}
			for i := range got {
				if !almostEqual(got[i], tc.want[i], 1e-9) {
					t.Errorf("Highs()[%d] = %v, want %v", i, got[i], tc.want[i])
				}
			}
		})
	}
}

func TestLows(t *testing.T) {
	tests := []struct {
		name string
		bars []domain.Bar
		want []float64
	}{
		{
			name: "empty",
			bars: nil,
			want: []float64{},
		},
		{
			name: "several bars",
			bars: []domain.Bar{
				mkBar(0, 10, 12, 9, 11),
				mkBar(1, 11, 13, 10, 12),
			},
			want: []float64{9, 10},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := Lows(tc.bars)
			if len(got) != len(tc.want) {
				t.Fatalf("Lows() len = %d, want %d", len(got), len(tc.want))
			}
			for i := range got {
				if !almostEqual(got[i], tc.want[i], 1e-9) {
					t.Errorf("Lows()[%d] = %v, want %v", i, got[i], tc.want[i])
				}
			}
		})
	}
}

func TestMaxN(t *testing.T) {
	tests := []struct {
		name   string
		xs     []float64
		n      int
		want   float64
		wantOk bool
	}{
		{
			name:   "empty",
			xs:     nil,
			n:      3,
			wantOk: false,
		},
		{
			name:   "exact length",
			xs:     []float64{5, 9, 2},
			n:      3,
			want:   9,
			wantOk: true,
		},
		{
			name: "longer than n uses last n only",
			xs:   []float64{100, 1, 2, 3},
			n:    3,
			// last 3: 1,2,3 -> max 3 (the leading 100 outside the window must
			// not be picked up)
			want:   3,
			wantOk: true,
		},
		{
			name:   "n<1 guard",
			xs:     []float64{1, 2, 3},
			n:      0,
			wantOk: false,
		},
		{
			name:   "n greater than length",
			xs:     []float64{1, 2},
			n:      3,
			wantOk: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := MaxN(tc.xs, tc.n)
			if ok != tc.wantOk {
				t.Fatalf("MaxN() ok = %v, want %v", ok, tc.wantOk)
			}
			if ok && !almostEqual(got, tc.want, 1e-9) {
				t.Errorf("MaxN() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestMinN(t *testing.T) {
	tests := []struct {
		name   string
		xs     []float64
		n      int
		want   float64
		wantOk bool
	}{
		{
			name:   "empty",
			xs:     nil,
			n:      3,
			wantOk: false,
		},
		{
			name:   "exact length",
			xs:     []float64{5, 9, 2},
			n:      3,
			want:   2,
			wantOk: true,
		},
		{
			name: "longer than n uses last n only",
			xs:   []float64{-100, 4, 5, 6},
			n:    3,
			// last 3: 4,5,6 -> min 4 (the leading -100 outside the window
			// must not be picked up)
			want:   4,
			wantOk: true,
		},
		{
			name:   "n<1 guard",
			xs:     []float64{1, 2, 3},
			n:      0,
			wantOk: false,
		},
		{
			name:   "n greater than length",
			xs:     []float64{1, 2},
			n:      3,
			wantOk: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := MinN(tc.xs, tc.n)
			if ok != tc.wantOk {
				t.Fatalf("MinN() ok = %v, want %v", ok, tc.wantOk)
			}
			if ok && !almostEqual(got, tc.want, 1e-9) {
				t.Errorf("MinN() = %v, want %v", got, tc.want)
			}
		})
	}
}
