package strategy

import (
	"math"

	"tradeforge/internal/domain"
)

// Convention: every indicator here returns (value, ok bool). ok is false
// when n <= 0 or the input is too short to compute the indicator; callers
// must check ok before using value (value is 0 when !ok, never NaN/Inf).

// SMA returns the simple moving average of the last n values in closes.
func SMA(closes []float64, n int) (float64, bool) {
	if n <= 0 || len(closes) < n {
		return 0, false
	}
	window := closes[len(closes)-n:]
	sum := 0.0
	for _, v := range window {
		sum += v
	}
	return sum / float64(n), true
}

// EMA returns the exponential moving average of closes with period n,
// seeded by an SMA of the first n values and applied forward across the
// remainder. Requires at least n values.
func EMA(closes []float64, n int) (float64, bool) {
	if n <= 0 || len(closes) < n {
		return 0, false
	}
	k := 2.0 / (float64(n) + 1.0)

	seed, ok := SMA(closes[:n], n)
	if !ok {
		return 0, false
	}

	ema := seed
	for _, v := range closes[n:] {
		ema = v*k + ema*(1-k)
	}
	return ema, true
}

// RSI returns the simple-average RSI (Cutler's RSI) over the last n periods
// of closes: average gain and average loss are plain arithmetic means of
// the last n price changes — NOT Wilder's recursive smoothing. This is the
// convention used by Connors RSI(2). Requires at least n+1 values (n price
// changes).
func RSI(closes []float64, n int) (float64, bool) {
	if n <= 0 || len(closes) < n+1 {
		return 0, false
	}

	// Most recent n+1 closes (n changes), averaged arithmetically per
	// Cutler's formulation.
	window := closes[len(closes)-(n+1):]

	var gainSum, lossSum float64
	for i := 1; i <= n; i++ {
		change := window[i] - window[i-1]
		if change > 0 {
			gainSum += change
		} else {
			lossSum += -change
		}
	}
	avgGain := gainSum / float64(n)
	avgLoss := lossSum / float64(n)

	if avgLoss == 0 {
		if avgGain == 0 {
			return 50, true // flat series: neutral RSI
		}
		return 100, true
	}

	rs := avgGain / avgLoss
	rsi := 100 - (100 / (1 + rs))
	return rsi, true
}

// ATR returns the Wilder Average True Range over the last n bars. Requires
// at least n+1 bars (needs a previous close for the first true range in the
// window).
func ATR(bars []domain.Bar, n int) (float64, bool) {
	if n <= 0 || len(bars) < n+1 {
		return 0, false
	}

	window := bars[len(bars)-(n+1):]
	var trSum float64
	for i := 1; i < len(window); i++ {
		trSum += trueRange(window[i], window[i-1])
	}
	return trSum / float64(n), true
}

func trueRange(cur, prev domain.Bar) float64 {
	hl := cur.High - cur.Low
	hc := math.Abs(cur.High - prev.Close)
	lc := math.Abs(cur.Low - prev.Close)
	return math.Max(hl, math.Max(hc, lc))
}

// StdDev returns the population standard deviation of the last n values in
// xs.
func StdDev(xs []float64, n int) (float64, bool) {
	if n <= 1 || len(xs) < n {
		return 0, false
	}
	window := xs[len(xs)-n:]

	mean := 0.0
	for _, v := range window {
		mean += v
	}
	mean /= float64(n)

	var sumSq float64
	for _, v := range window {
		d := v - mean
		sumSq += d * d
	}
	return math.Sqrt(sumSq / float64(n)), true
}

// Closes extracts the Close price from a slice of bars, oldest first.
func Closes(bars []domain.Bar) []float64 {
	out := make([]float64, len(bars))
	for i, b := range bars {
		out[i] = b.Close
	}
	return out
}

// Highs extracts the High price from a slice of bars, oldest first.
func Highs(bars []domain.Bar) []float64 {
	out := make([]float64, len(bars))
	for i, b := range bars {
		out[i] = b.High
	}
	return out
}

// Lows extracts the Low price from a slice of bars, oldest first.
func Lows(bars []domain.Bar) []float64 {
	out := make([]float64, len(bars))
	for i, b := range bars {
		out[i] = b.Low
	}
	return out
}

// MaxN returns the maximum of the last n values in xs. ok is false when
// n < 1 or len(xs) < n.
func MaxN(xs []float64, n int) (float64, bool) {
	if n < 1 || len(xs) < n {
		return 0, false
	}
	window := xs[len(xs)-n:]
	max := window[0]
	for _, v := range window[1:] {
		if v > max {
			max = v
		}
	}
	return max, true
}

// MinN returns the minimum of the last n values in xs. ok is false when
// n < 1 or len(xs) < n.
func MinN(xs []float64, n int) (float64, bool) {
	if n < 1 || len(xs) < n {
		return 0, false
	}
	window := xs[len(xs)-n:]
	min := window[0]
	for _, v := range window[1:] {
		if v < min {
			min = v
		}
	}
	return min, true
}
