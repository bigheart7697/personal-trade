// Deflated Sharpe ratio (DSR) machinery.
//
// Reference: Bailey, D. H. and López de Prado, M., "The Deflated Sharpe
// Ratio: Correcting for Selection Bias, Backtest Overfitting, and
// Non-Normality" (2014), Journal of Portfolio Management.
//
// Honesty caveat: DSR corrects for the number of independent trials that
// competed to produce the reported Sharpe ratio. In this codebase, the
// walk-forward harness's trials come from overlapping/rolling train windows
// across folds, so they are not fully statistically independent — pooling
// them and treating the pool as independent trials (as BLP's practical
// applications typically do) is the standard approximation used here, not
// an exact test. Treat "DSR >= 0.95" as strong evidence against pure
// selection-bias luck, not as a rigorous p-value.
package metrics

import "math"

// eulerMascheroni is the Euler-Mascheroni constant (gamma_E), used in the
// expected-value-of-the-maximum-of-N-Sharpe-ratios approximation.
const eulerMascheroni = 0.57721566490153286060

// Skewness returns the population skewness of xs: (1/n)*sum((x-mean)^3) /
// stddev^3, where stddev is the POPULATION standard deviation (n divisor,
// not n-1). Returns 0 if n < 3 or the population standard deviation is 0
// (no meaningful shape to report).
func Skewness(xs []float64) float64 {
	n := len(xs)
	if n < 3 {
		return 0
	}
	m := mean(xs)
	var sumSq, sumCube float64
	for _, x := range xs {
		d := x - m
		sumSq += d * d
		sumCube += d * d * d
	}
	sp := math.Sqrt(sumSq / float64(n))
	if sp == 0 {
		return 0
	}
	return (sumCube / float64(n)) / (sp * sp * sp)
}

// Kurtosis returns the population, NON-excess kurtosis of xs:
// (1/n)*sum((x-mean)^4) / stddev^4, where stddev is the POPULATION standard
// deviation (n divisor). A normal distribution has kurtosis ~= 3. Returns 3
// (the neutral, normal-distribution default) if n < 4 or the population
// standard deviation is 0 — there isn't enough information to say the
// distribution is anything other than normal-shaped, and 3 makes
// ProbabilisticSharpe's kurt-adjustment term vanish.
func Kurtosis(xs []float64) float64 {
	n := len(xs)
	if n < 4 {
		return 3
	}
	m := mean(xs)
	var sumSq, sumQuad float64
	for _, x := range xs {
		d := x - m
		sumSq += d * d
		sumQuad += d * d * d * d
	}
	sp := math.Sqrt(sumSq / float64(n))
	if sp == 0 {
		return 3
	}
	return (sumQuad / float64(n)) / (sp * sp * sp * sp)
}

// NormalCDF returns the standard normal cumulative distribution function
// Phi(z), computed via the complementary error function for numerical
// stability.
func NormalCDF(z float64) float64 {
	return 0.5 * math.Erfc(-z/math.Sqrt2)
}

// InverseNormalCDF returns the standard normal quantile function Phi^-1(p)
// (the probit), using Acklam's rational approximation (relative error <
// 1.15e-9 over the full domain). ok is false for p <= 0 or p >= 1, where the
// probit is undefined (+/-Infinity).
func InverseNormalCDF(p float64) (float64, bool) {
	if p <= 0 || p >= 1 {
		return 0, false
	}

	// Coefficients for the rational approximations, as published by Peter
	// J. Acklam.
	const (
		a1 = -3.969683028665376e+01
		a2 = 2.209460984245205e+02
		a3 = -2.759285104469687e+02
		a4 = 1.383577518672690e+02
		a5 = -3.066479806614716e+01
		a6 = 2.506628277459239e+00

		b1 = -5.447609879822406e+01
		b2 = 1.615858368580409e+02
		b3 = -1.556989798598866e+02
		b4 = 6.680131188771972e+01
		b5 = -1.328068155288572e+01

		c1 = -7.784894002430293e-03
		c2 = -3.223964580411365e-01
		c3 = -2.400758277161838e+00
		c4 = -2.549732539343734e+00
		c5 = 4.374664141464968e+00
		c6 = 2.938163982698783e+00

		d1 = 7.784695709041462e-03
		d2 = 3.224671290700398e-01
		d3 = 2.445134137142996e+00
		d4 = 3.754408661907416e+00
	)

	const pLow = 0.02425
	pHigh := 1 - pLow

	var x float64
	switch {
	case p < pLow:
		// Rational approximation for the lower region.
		q := math.Sqrt(-2 * math.Log(p))
		x = (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q + c6) /
			((((d1*q+d2)*q+d3)*q+d4)*q + 1)
	case p <= pHigh:
		// Rational approximation for the central region.
		q := p - 0.5
		r := q * q
		x = (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r + a6) * q /
			(((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r + 1)
	default:
		// Rational approximation for the upper region.
		q := math.Sqrt(-2 * math.Log(1-p))
		x = -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q + c6) /
			((((d1*q+d2)*q+d3)*q+d4)*q + 1)
	}

	// One step of Halley's rational method refines the approximation to
	// full float64 precision.
	e := NormalCDF(x) - p
	u := e * math.Sqrt(2*math.Pi) * math.Exp(x*x/2)
	x = x - u/(1+x*u/2)

	return x, true
}

// ProbabilisticSharpe returns PSR(srBench), the probability that the true
// Sharpe ratio exceeds srBench given an observed (sample) Sharpe ratio
// srHat computed from n observations with skewness skew and (non-excess)
// kurtosis kurt:
//
//	z = (srHat - srBench) * sqrt(n-1) / sqrt(1 - skew*srHat + ((kurt-1)/4)*srHat^2)
//	PSR = NormalCDF(z)
//
// ok is false when n < 2 or the radicand under the denominator's square
// root is <= 0 (degenerate — the PSR is not well-defined there).
func ProbabilisticSharpe(srHat, srBench float64, n int, skew, kurt float64) (float64, bool) {
	if n < 2 {
		return 0, false
	}
	radicand := 1 - skew*srHat + ((kurt-1)/4)*srHat*srHat
	if radicand <= 0 {
		return 0, false
	}
	z := (srHat - srBench) * math.Sqrt(float64(n-1)) / math.Sqrt(radicand)
	return NormalCDF(z), true
}

// ExpectedMaxSharpe approximates E[max Sharpe ratio across nTrials
// independent trials], given the variance of the Sharpe ratios across those
// trials (trialSRVar), per Bailey & Lopez de Prado (2014):
//
//	sqrt(trialSRVar) * ((1-gammaE)*Phi^-1(1-1/N) + gammaE*Phi^-1(1-1/(N*e)))
//
// where N = nTrials and gammaE is the Euler-Mascheroni constant. Returns 0
// for nTrials <= 1 or trialSRVar <= 0, where "the best of N trials" is not
// a meaningful correction (nothing to select among, or no dispersion to
// exploit by selection).
func ExpectedMaxSharpe(nTrials int, trialSRVar float64) float64 {
	if nTrials <= 1 || trialSRVar <= 0 {
		return 0
	}
	n := float64(nTrials)

	// Both quantile arguments are in (0,1) and in-domain for N >= 2:
	// 1 - 1/N ranges over [0.5, 1) and 1 - 1/(N*e) ranges over [1-1/e, 1).
	q1, ok1 := InverseNormalCDF(1 - 1/n)
	q2, ok2 := InverseNormalCDF(1 - 1/(n*math.E))
	if !ok1 || !ok2 {
		return 0
	}

	return math.Sqrt(trialSRVar) * ((1-eulerMascheroni)*q1 + eulerMascheroni*q2)
}

// DeflatedSharpe computes the deflated Sharpe ratio of a daily return
// series, correcting the observed Sharpe ratio for the number of trials
// (nTrials) that competed to produce it and the dispersion of Sharpe ratios
// across those trials (trialSRVar). DSR is a probability in [0,1]; values
// close to 1 indicate the observed Sharpe is unlikely to be a
// selection-bias artifact of trying many parameter combinations. The
// project's promotion gate (docs/ROADMAP.md, G1) reads "deflated Sharpe p <
// 0.05", equivalently DSR >= 0.95.
//
// ok is false when dailyReturns has fewer than 2 points, has zero sample
// standard deviation (degenerate — no Sharpe ratio is defined), or when the
// underlying ProbabilisticSharpe computation is degenerate.
func DeflatedSharpe(dailyReturns []float64, nTrials int, trialSRVar float64) (dsr float64, ok bool) {
	n := len(dailyReturns)
	if n < 2 {
		return 0, false
	}

	m := mean(dailyReturns)
	s := stdDev(dailyReturns) // sample stddev, n-1 divisor
	if s == 0 {
		return 0, false
	}

	srHat := m / s
	skew := Skewness(dailyReturns)
	kurt := Kurtosis(dailyReturns)

	sr0 := ExpectedMaxSharpe(nTrials, trialSRVar)

	return ProbabilisticSharpe(srHat, sr0, n, skew, kurt)
}
