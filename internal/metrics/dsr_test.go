package metrics

import (
	"math"
	"testing"
)

func TestSkewness(t *testing.T) {
	tests := []struct {
		name string
		xs   []float64
		want float64
		tol  float64
	}{
		{name: "n<3 returns 0", xs: []float64{1, 2}, want: 0, tol: 1e-9},
		{name: "zero variance returns 0", xs: []float64{5, 5, 5, 5}, want: 0, tol: 1e-9},
		{name: "symmetric data has zero skew", xs: []float64{-2, -1, 0, 1, 2}, want: 0, tol: 1e-9},
		{name: "symmetric data 6 elements", xs: []float64{1, 2, 3, 4, 5, 6}, want: 0, tol: 1e-9},
		// Hand-verified asymmetric case: {1,2,3,4,10}.
		// mean = 4; deviations = -3,-2,-1,0,6; sumSq=9+4+1+0+36=50; sp=sqrt(50/5)=sqrt(10)
		// sumCube = -27-8-1+0+216 = 180; skew = (180/5) / (sqrt(10))^3 = 36 / 31.6227766 = 1.1384199577
		{name: "hand-verified asymmetric", xs: []float64{1, 2, 3, 4, 10}, want: 1.1384199577, tol: 1e-6},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := Skewness(tc.xs)
			if !almostEqual(got, tc.want, tc.tol) {
				t.Errorf("Skewness(%v) = %v, want %v", tc.xs, got, tc.want)
			}
		})
	}
}

func TestKurtosis(t *testing.T) {
	tests := []struct {
		name string
		xs   []float64
		want float64
		tol  float64
	}{
		{name: "n<4 returns neutral 3", xs: []float64{1, 2, 3}, want: 3, tol: 1e-9},
		{name: "zero variance returns neutral 3", xs: []float64{5, 5, 5, 5}, want: 3, tol: 1e-9},
		// Hand-verified: {1,2,3,4,5,6,7}. mean=4; deviations=-3,-2,-1,0,1,2,3
		// sumSq = 9+4+1+0+1+4+9=28; sp = sqrt(28/7)=sqrt(4)=2
		// sumQuad = 81+16+1+0+1+16+81=196; kurt=(196/7)/(2^4)=28/16=1.75
		{name: "hand-verified uniform-ish", xs: []float64{1, 2, 3, 4, 5, 6, 7}, want: 1.75, tol: 1e-9},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := Kurtosis(tc.xs)
			if !almostEqual(got, tc.want, tc.tol) {
				t.Errorf("Kurtosis(%v) = %v, want %v", tc.xs, got, tc.want)
			}
		})
	}
}

func TestNormalCDF(t *testing.T) {
	tests := []struct {
		name string
		z    float64
		want float64
		tol  float64
	}{
		{name: "z=0", z: 0, want: 0.5, tol: 1e-9},
		{name: "z=1.959964 -> ~0.975", z: 1.959964, want: 0.975, tol: 1e-6},
		{name: "z=-1.959964 -> ~0.025", z: -1.959964, want: 0.025, tol: 1e-6},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := NormalCDF(tc.z)
			if !almostEqual(got, tc.want, tc.tol) {
				t.Errorf("NormalCDF(%v) = %v, want %v", tc.z, got, tc.want)
			}
		})
	}
}

func TestInverseNormalCDF(t *testing.T) {
	t.Run("Phi^-1(0.5) = 0", func(t *testing.T) {
		got, ok := InverseNormalCDF(0.5)
		if !ok {
			t.Fatal("ok = false, want true")
		}
		if !almostEqual(got, 0, 1e-9) {
			t.Errorf("InverseNormalCDF(0.5) = %v, want 0", got)
		}
	})

	t.Run("Phi^-1(0.975) = 1.959963985", func(t *testing.T) {
		got, ok := InverseNormalCDF(0.975)
		if !ok {
			t.Fatal("ok = false, want true")
		}
		if !almostEqual(got, 1.959963985, 1e-6) {
			t.Errorf("InverseNormalCDF(0.975) = %v, want 1.959963985", got)
		}
	})

	t.Run("Phi^-1(0.025) = -1.959963985", func(t *testing.T) {
		got, ok := InverseNormalCDF(0.025)
		if !ok {
			t.Fatal("ok = false, want true")
		}
		if !almostEqual(got, -1.959963985, 1e-6) {
			t.Errorf("InverseNormalCDF(0.025) = %v, want -1.959963985", got)
		}
	})

	t.Run("Phi^-1(0.999) = 3.090232", func(t *testing.T) {
		got, ok := InverseNormalCDF(0.999)
		if !ok {
			t.Fatal("ok = false, want true")
		}
		if !almostEqual(got, 3.090232, 1e-4) {
			t.Errorf("InverseNormalCDF(0.999) = %v, want 3.090232", got)
		}
	})

	t.Run("symmetry Phi^-1(p) = -Phi^-1(1-p)", func(t *testing.T) {
		ps := []float64{0.001, 0.01, 0.1, 0.3, 0.4, 0.6, 0.9, 0.99, 0.999}
		for _, p := range ps {
			a, ok1 := InverseNormalCDF(p)
			b, ok2 := InverseNormalCDF(1 - p)
			if !ok1 || !ok2 {
				t.Fatalf("p=%v: ok1=%v ok2=%v, want both true", p, ok1, ok2)
			}
			if !almostEqual(a, -b, 1e-9) {
				t.Errorf("p=%v: InverseNormalCDF(p)=%v, -InverseNormalCDF(1-p)=%v, want equal", p, a, -b)
			}
		}
	})

	t.Run("out of domain", func(t *testing.T) {
		cases := []float64{0, -0.5, 1, 1.5}
		for _, p := range cases {
			_, ok := InverseNormalCDF(p)
			if ok {
				t.Errorf("InverseNormalCDF(%v) ok = true, want false", p)
			}
		}
	})
}

func TestProbabilisticSharpe(t *testing.T) {
	t.Run("srHat >> bench with large n approaches 1", func(t *testing.T) {
		got, ok := ProbabilisticSharpe(2.0, 0.0, 1000, 0, 3)
		if !ok {
			t.Fatal("ok = false, want true")
		}
		if got < 0.999 {
			t.Errorf("ProbabilisticSharpe = %v, want close to 1", got)
		}
	})

	t.Run("srHat == bench with skew 0 kurt 3 gives 0.5", func(t *testing.T) {
		got, ok := ProbabilisticSharpe(1.0, 1.0, 100, 0, 3)
		if !ok {
			t.Fatal("ok = false, want true")
		}
		if !almostEqual(got, 0.5, 1e-9) {
			t.Errorf("ProbabilisticSharpe = %v, want 0.5", got)
		}
	})

	t.Run("n < 2 is degenerate", func(t *testing.T) {
		_, ok := ProbabilisticSharpe(1.0, 0.0, 1, 0, 3)
		if ok {
			t.Error("ok = true, want false for n < 2")
		}
	})

	t.Run("non-positive radicand is degenerate", func(t *testing.T) {
		// radicand = 1 - skew*srHat + ((kurt-1)/4)*srHat^2 = 1 - 1*5 + 0*25 = -4 <= 0.
		_, ok := ProbabilisticSharpe(5.0, 0.0, 100, 1.0, 1)
		if ok {
			t.Error("ok = true, want false for non-positive radicand")
		}
	})
}

func TestExpectedMaxSharpe(t *testing.T) {
	t.Run("nTrials <= 1 returns 0", func(t *testing.T) {
		if got := ExpectedMaxSharpe(1, 0.5); got != 0 {
			t.Errorf("ExpectedMaxSharpe(1, 0.5) = %v, want 0", got)
		}
		if got := ExpectedMaxSharpe(0, 0.5); got != 0 {
			t.Errorf("ExpectedMaxSharpe(0, 0.5) = %v, want 0", got)
		}
	})

	t.Run("trialSRVar <= 0 returns 0", func(t *testing.T) {
		if got := ExpectedMaxSharpe(10, 0); got != 0 {
			t.Errorf("ExpectedMaxSharpe(10, 0) = %v, want 0", got)
		}
		if got := ExpectedMaxSharpe(10, -1); got != 0 {
			t.Errorf("ExpectedMaxSharpe(10, -1) = %v, want 0", got)
		}
	})

	t.Run("grows with N", func(t *testing.T) {
		v10 := ExpectedMaxSharpe(10, 0.1)
		v100 := ExpectedMaxSharpe(100, 0.1)
		v1000 := ExpectedMaxSharpe(1000, 0.1)
		if !(v10 < v100 && v100 < v1000) {
			t.Errorf("ExpectedMaxSharpe not increasing in N: v10=%v v100=%v v1000=%v", v10, v100, v1000)
		}
	})

	t.Run("grows with trialSRVar", func(t *testing.T) {
		vLow := ExpectedMaxSharpe(100, 0.01)
		vHigh := ExpectedMaxSharpe(100, 1.0)
		if !(vLow < vHigh) {
			t.Errorf("ExpectedMaxSharpe not increasing in trialSRVar: vLow=%v vHigh=%v", vLow, vHigh)
		}
	})
}

func TestDeflatedSharpe_Monotonicity(t *testing.T) {
	// Identical daily returns; only nTrials rises. More trials -> higher
	// expected max Sharpe under the null -> DSR should be non-increasing.
	returns := make([]float64, 0, 300)
	for i := 0; i < 300; i++ {
		// Mildly noisy positive-drift series so skew/kurt aren't degenerate.
		v := 0.001 + 0.01*math.Sin(float64(i)*0.37)
		returns = append(returns, v)
	}

	const trialSRVar = 0.05
	trialsSeq := []int{1, 10, 100}
	var prev float64
	var prevOk bool
	for i, nTrials := range trialsSeq {
		dsr, ok := DeflatedSharpe(returns, nTrials, trialSRVar)
		if !ok {
			t.Fatalf("nTrials=%d: DeflatedSharpe ok = false, want true", nTrials)
		}
		if dsr < 0 || dsr > 1 {
			t.Fatalf("nTrials=%d: DSR = %v, want in [0,1]", nTrials, dsr)
		}
		if i > 0 && prevOk && dsr > prev+1e-12 {
			t.Errorf("nTrials=%d: DSR = %v > previous DSR = %v, want non-increasing", nTrials, dsr, prev)
		}
		prev = dsr
		prevOk = ok
	}
}

func TestDeflatedSharpe_Degenerate(t *testing.T) {
	t.Run("constant returns", func(t *testing.T) {
		returns := []float64{0.01, 0.01, 0.01, 0.01, 0.01}
		_, ok := DeflatedSharpe(returns, 10, 0.05)
		if ok {
			t.Error("ok = true, want false for zero-variance returns")
		}
	})

	t.Run("n < 2", func(t *testing.T) {
		_, ok := DeflatedSharpe([]float64{0.01}, 10, 0.05)
		if ok {
			t.Error("ok = true, want false for n < 2")
		}
	})

	t.Run("empty", func(t *testing.T) {
		_, ok := DeflatedSharpe(nil, 10, 0.05)
		if ok {
			t.Error("ok = true, want false for empty input")
		}
	})
}

func TestDeflatedSharpe_NoNaNInf(t *testing.T) {
	returns := make([]float64, 0, 100)
	for i := 0; i < 100; i++ {
		v := 0.002 + 0.02*math.Sin(float64(i)*0.11) - 0.01*math.Cos(float64(i)*0.53)
		returns = append(returns, v)
	}
	dsr, ok := DeflatedSharpe(returns, 144, 0.08)
	if !ok {
		t.Fatal("ok = false, want true")
	}
	if math.IsNaN(dsr) || math.IsInf(dsr, 0) {
		t.Errorf("DSR = %v, want finite", dsr)
	}
	if dsr < 0 || dsr > 1 {
		t.Errorf("DSR = %v, want in [0,1]", dsr)
	}
}
