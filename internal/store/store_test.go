package store

import (
	"path/filepath"
	"testing"
	"time"

	"tradeforge/internal/backtest"
	"tradeforge/internal/domain"
	"tradeforge/internal/eval"
	"tradeforge/internal/metrics"
	"tradeforge/internal/strategy"
)

func openTestStore(t *testing.T) *Store {
	t.Helper()
	path := filepath.Join(t.TempDir(), "test.db")
	s, err := Open(path)
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Errorf("Close() error = %v", err)
		}
	})
	return s
}

func fakeBacktestResult() backtest.Result {
	return backtest.Result{
		StrategyName: "sma-cross",
		Horizon:      strategy.Long,
		PeriodStart:  time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC),
		PeriodEnd:    time.Date(2020, 12, 31, 0, 0, 0, 0, time.UTC),
		Fills: []domain.Fill{
			{Order: domain.Order{Symbol: "SPY", Side: domain.Buy, Qty: 10}, Price: 100},
			{Order: domain.Order{Symbol: "SPY", Side: domain.Sell, Qty: 10}, Price: 110},
		},
		Metrics: metrics.Metrics{
			TotalReturn: 0.1,
			CAGR:        0.1,
			Sharpe:      1.23,
			MaxDrawdown: 0.05,
			NumTrades:   2,
		},
		EquityCurve: fakeEquityCurve(),
	}
}

func fakeEquityCurve() []metrics.EquityPoint {
	return []metrics.EquityPoint{
		{Time: time.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC), Equity: 100000},
		{Time: time.Date(2018, 1, 2, 0, 0, 0, 0, time.UTC), Equity: 101000},
		{Time: time.Date(2018, 1, 3, 0, 0, 0, 0, time.UTC), Equity: 99500},
	}
}

func fakeWalkForwardResult() eval.Result {
	return eval.Result{
		StrategyName: "rsi2",
		Horizon:      strategy.Short,
		Symbol:       "SPY",
		TrainBars:    756,
		TestBars:     252,
		OOSEquity:    fakeEquityCurve(),
		Folds: []eval.FoldResult{
			{
				Fold:           1,
				TrainStart:     time.Date(2015, 1, 1, 0, 0, 0, 0, time.UTC),
				TrainEnd:       time.Date(2017, 12, 31, 0, 0, 0, 0, time.UTC),
				TestStart:      time.Date(2018, 1, 1, 0, 0, 0, 0, time.UTC),
				TestEnd:        time.Date(2018, 12, 31, 0, 0, 0, 0, time.UTC),
				BestParams:     map[string]float64{"entryRSI": 5, "exitRSI": 60, "timeStop": 10},
				TrainObjective: 1.1,
				TestMetrics:    metrics.Metrics{Sharpe: 0.9, MaxDrawdown: 0.08},
				NumFills:       4,
				Trials:         18,
			},
			{
				Fold:           2,
				TrainStart:     time.Date(2016, 1, 1, 0, 0, 0, 0, time.UTC),
				TrainEnd:       time.Date(2018, 12, 31, 0, 0, 0, 0, time.UTC),
				TestStart:      time.Date(2019, 1, 1, 0, 0, 0, 0, time.UTC),
				TestEnd:        time.Date(2019, 12, 31, 0, 0, 0, 0, time.UTC),
				BestParams:     map[string]float64{"entryRSI": 10, "exitRSI": 70, "timeStop": 5},
				TrainObjective: 1.4,
				TestMetrics:    metrics.Metrics{Sharpe: 1.5, MaxDrawdown: 0.03},
				NumFills:       6,
				Trials:         18,
			},
		},
		OOSMetrics:       metrics.Metrics{Sharpe: 1.2, MaxDrawdown: 0.08, CAGR: 0.15},
		BenchmarkMetrics: metrics.Metrics{Sharpe: 0.6, MaxDrawdown: 0.12, CAGR: 0.09},
		TotalTrials:      36,
		TotalFills:       10,
		TrialSRVar:       0.02,
		DSR:              0.87,
		DSROk:            true,
		Regimes:          fakeRegimeSlices(),
		RegimeDropped:    3,
		RandomBase:       fakeRandomBaseline(),
	}
}

func fakeRegimeSlices() []eval.RegimeSlice {
	return []eval.RegimeSlice{
		{Regime: eval.Bull, Days: 100, Share: 0.5, StratAnnRet: 0.2, StratSharpe: 1.1, StratWorst: -0.02, BenchAnnRet: 0.15, BenchSharpe: 0.9, BenchWorst: -0.03},
		{Regime: eval.Bear, Days: 60, Share: 0.3, StratAnnRet: -0.05, StratSharpe: -0.3, StratWorst: -0.08, BenchAnnRet: -0.25, BenchSharpe: -1.2, BenchWorst: -0.1},
		{Regime: eval.Chop, Days: 40, Share: 0.2, StratAnnRet: 0.01, StratSharpe: 0.1, StratWorst: -0.01, BenchAnnRet: 0.0, BenchSharpe: 0.0, BenchWorst: -0.015},
	}
}

func fakeRandomBaseline() eval.RandomBaseline {
	return eval.RandomBaseline{
		Trials:       200,
		Blocks:       12,
		BlockLen:     15,
		MedianSharpe: 0.31,
		P95Sharpe:    0.74,
		StrategyPct:  0.82,
		Beats:        false,
		OK:           true,
	}
}

func fakeKFoldResult() eval.KFoldResult {
	return eval.KFoldResult{
		StrategyName: "rsi2",
		Horizon:      strategy.Short,
		Symbol:       "SPY",
		Folds:        4,
		PurgeBars:    20,
		EmbargoBars:  6,
		OOSEquity:    fakeEquityCurve(),
		FoldResults: []eval.KFoldFold{
			{
				Fold:           1,
				TestStart:      time.Date(2015, 1, 1, 0, 0, 0, 0, time.UTC),
				TestEnd:        time.Date(2015, 12, 31, 0, 0, 0, 0, time.UTC),
				BestParams:     map[string]float64{"entryRSI": 5, "exitRSI": 60},
				TrainObjective: 1.1,
				TestMetrics:    metrics.Metrics{Sharpe: 0.9, MaxDrawdown: 0.08},
				NumFills:       4,
				Trials:         18,
			},
			{
				Fold:           2,
				TestStart:      time.Date(2016, 1, 1, 0, 0, 0, 0, time.UTC),
				TestEnd:        time.Date(2016, 12, 31, 0, 0, 0, 0, time.UTC),
				BestParams:     map[string]float64{"entryRSI": 10, "exitRSI": 70},
				TrainObjective: 1.4,
				TestMetrics:    metrics.Metrics{Sharpe: 1.5, MaxDrawdown: 0.03},
				NumFills:       6,
				Trials:         18,
			},
		},
		ParamWins:        map[string]int{"entryRSI=5 exitRSI=60": 1, "entryRSI=10 exitRSI=70": 1},
		OOSMetrics:       metrics.Metrics{Sharpe: 1.2, MaxDrawdown: 0.08, CAGR: 0.15},
		BenchmarkMetrics: metrics.Metrics{Sharpe: 0.6, MaxDrawdown: 0.12, CAGR: 0.09},
		TotalTrials:      36,
		TotalFills:       10,
		TrialSRVar:       0.02,
		DSR:              0.87,
		DSROk:            true,
		Regimes:          fakeRegimeSlices(),
		RegimeDropped:    2,
		RandomBase:       fakeRandomBaseline(),
	}
}

func TestSaveBacktest_RoundTrip(t *testing.T) {
	s := openTestStore(t)

	meta := RunMeta{
		Strategy: "sma-cross",
		Horizon:  "long-term",
		Symbol:   "SPY",
		DataPath: "data/SPY.csv",
		Config:   Config{Cash: 100000, SlippageBps: 5},
	}
	res := fakeBacktestResult()

	id, err := s.SaveBacktest(meta, res)
	if err != nil {
		t.Fatalf("SaveBacktest() error = %v", err)
	}
	if id <= 0 {
		t.Fatalf("SaveBacktest() id = %d, want > 0", id)
	}

	detail, err := s.GetRun(id)
	if err != nil {
		t.Fatalf("GetRun() error = %v", err)
	}

	if detail.Kind != "backtest" {
		t.Errorf("Kind = %q, want backtest", detail.Kind)
	}
	if detail.Strategy != "sma-cross" || detail.Horizon != "long-term" || detail.Symbol != "SPY" {
		t.Errorf("Strategy/Horizon/Symbol = %q/%q/%q, want sma-cross/long-term/SPY",
			detail.Strategy, detail.Horizon, detail.Symbol)
	}
	if detail.DataPath != "data/SPY.csv" {
		t.Errorf("DataPath = %q, want data/SPY.csv", detail.DataPath)
	}
	if detail.PeriodStart != "2020-01-01" || detail.PeriodEnd != "2020-12-31" {
		t.Errorf("PeriodStart/End = %q/%q, want 2020-01-01/2020-12-31", detail.PeriodStart, detail.PeriodEnd)
	}
	if detail.Trials != 0 {
		t.Errorf("Trials = %d, want 0", detail.Trials)
	}
	if detail.Fills != 2 {
		t.Errorf("Fills = %d, want 2", detail.Fills)
	}
	if detail.Metrics.Sharpe != 1.23 {
		t.Errorf("Metrics.Sharpe = %v, want 1.23", detail.Metrics.Sharpe)
	}
	if detail.Config.Cash != 100000 || detail.Config.SlippageBps != 5 {
		t.Errorf("Config = %+v, want Cash=100000 SlippageBps=5", detail.Config)
	}
	if len(detail.Folds) != 0 {
		t.Errorf("Folds = %d, want 0 for a plain backtest", len(detail.Folds))
	}
	if detail.TrialSRVar != 0 || detail.DSR != 0 {
		t.Errorf("TrialSRVar/DSR = %v/%v, want 0/0 for a plain backtest", detail.TrialSRVar, detail.DSR)
	}
	if len(detail.Equity) != 3 {
		t.Fatalf("len(Equity) = %d, want 3", len(detail.Equity))
	}
	if detail.Equity[0].T != "2018-01-01" || detail.Equity[0].V != 100000 {
		t.Errorf("Equity[0] = %+v, want {2018-01-01 100000}", detail.Equity[0])
	}
	if detail.Benchmark != nil {
		t.Errorf("Benchmark = %+v, want nil for a plain backtest", detail.Benchmark)
	}
	if detail.Regimes != nil {
		t.Errorf("Regimes = %+v, want nil for a plain backtest", detail.Regimes)
	}
	if detail.RandomBase != nil {
		t.Errorf("RandomBase = %+v, want nil for a plain backtest", detail.RandomBase)
	}
}

func TestSaveWalkForward_RoundTrip(t *testing.T) {
	s := openTestStore(t)

	meta := RunMeta{
		Strategy: "rsi2",
		Horizon:  "short-term",
		Symbol:   "SPY",
		DataPath: "data/SPY.csv",
		Config:   Config{Cash: 100000, SlippageBps: 5, TrainBars: 756, TestBars: 252},
	}
	res := fakeWalkForwardResult()

	id, err := s.SaveWalkForward(meta, res)
	if err != nil {
		t.Fatalf("SaveWalkForward() error = %v", err)
	}

	detail, err := s.GetRun(id)
	if err != nil {
		t.Fatalf("GetRun() error = %v", err)
	}

	if detail.Kind != "walkforward" {
		t.Errorf("Kind = %q, want walkforward", detail.Kind)
	}
	if detail.Trials != 36 {
		t.Errorf("Trials = %d, want 36 (TotalTrials)", detail.Trials)
	}
	if detail.Fills != 10 {
		t.Errorf("Fills = %d, want 10 (TotalFills)", detail.Fills)
	}
	// metrics_json for a walkforward run holds the OOS metrics, not any
	// individual fold's.
	if detail.Metrics.Sharpe != 1.2 || detail.Metrics.CAGR != 0.15 {
		t.Errorf("Metrics = %+v, want OOS Sharpe=1.2 CAGR=0.15", detail.Metrics)
	}

	if len(detail.Folds) != 2 {
		t.Fatalf("len(Folds) = %d, want 2", len(detail.Folds))
	}

	f0 := detail.Folds[0]
	if f0.Fold != 1 {
		t.Errorf("Folds[0].Fold = %d, want 1", f0.Fold)
	}
	if f0.TrainStart != "2015-01-01" || f0.TestEnd != "2018-12-31" {
		t.Errorf("Folds[0] TrainStart/TestEnd = %q/%q, want 2015-01-01/2018-12-31", f0.TrainStart, f0.TestEnd)
	}
	if f0.BestParams["entryRSI"] != 5 || f0.BestParams["exitRSI"] != 60 || f0.BestParams["timeStop"] != 10 {
		t.Errorf("Folds[0].BestParams = %+v, want entryRSI=5 exitRSI=60 timeStop=10", f0.BestParams)
	}
	if f0.TrainObjective != 1.1 {
		t.Errorf("Folds[0].TrainObjective = %v, want 1.1", f0.TrainObjective)
	}
	if f0.TestMetrics.Sharpe != 0.9 {
		t.Errorf("Folds[0].TestMetrics.Sharpe = %v, want 0.9", f0.TestMetrics.Sharpe)
	}
	if f0.Fills != 4 || f0.Trials != 18 {
		t.Errorf("Folds[0].Fills/Trials = %d/%d, want 4/18", f0.Fills, f0.Trials)
	}

	f1 := detail.Folds[1]
	if f1.Fold != 2 {
		t.Errorf("Folds[1].Fold = %d, want 2", f1.Fold)
	}
	if f1.BestParams["entryRSI"] != 10 {
		t.Errorf("Folds[1].BestParams[entryRSI] = %v, want 10", f1.BestParams["entryRSI"])
	}

	if detail.TrialSRVar != 0.02 {
		t.Errorf("TrialSRVar = %v, want 0.02", detail.TrialSRVar)
	}
	if detail.DSR != 0.87 {
		t.Errorf("DSR = %v, want 0.87", detail.DSR)
	}

	if len(detail.Equity) != 3 {
		t.Fatalf("len(Equity) = %d, want 3", len(detail.Equity))
	}
	if detail.Equity[1].T != "2018-01-02" || detail.Equity[1].V != 101000 {
		t.Errorf("Equity[1] = %+v, want {2018-01-02 101000}", detail.Equity[1])
	}
	if detail.Benchmark == nil {
		t.Fatal("Benchmark = nil, want non-nil for a walkforward run")
	}
	if detail.Benchmark.Sharpe != 0.6 || detail.Benchmark.CAGR != 0.09 {
		t.Errorf("Benchmark = %+v, want Sharpe=0.6 CAGR=0.09", detail.Benchmark)
	}

	if detail.Regimes == nil {
		t.Fatal("Regimes = nil, want non-nil for a walkforward run")
	}
	if len(detail.Regimes.Slices) != 3 {
		t.Fatalf("len(Regimes.Slices) = %d, want 3", len(detail.Regimes.Slices))
	}
	if detail.Regimes.Slices[0].Regime != eval.Bull || detail.Regimes.Slices[0].Days != 100 {
		t.Errorf("Regimes.Slices[0] = %+v, want Bull/Days=100", detail.Regimes.Slices[0])
	}
	if detail.Regimes.Slices[1].Regime != eval.Bear || detail.Regimes.Slices[1].StratSharpe != -0.3 {
		t.Errorf("Regimes.Slices[1] = %+v, want Bear/StratSharpe=-0.3", detail.Regimes.Slices[1])
	}
	if detail.Regimes.Dropped != 3 {
		t.Errorf("Regimes.Dropped = %d, want 3", detail.Regimes.Dropped)
	}

	if detail.RandomBase == nil {
		t.Fatal("RandomBase = nil, want non-nil for a walkforward run")
	}
	if detail.RandomBase.Trials != 200 || detail.RandomBase.MedianSharpe != 0.31 || detail.RandomBase.P95Sharpe != 0.74 {
		t.Errorf("RandomBase = %+v, want Trials=200 MedianSharpe=0.31 P95Sharpe=0.74", detail.RandomBase)
	}
	if detail.RandomBase.StrategyPct != 0.82 || detail.RandomBase.Beats != false || !detail.RandomBase.OK {
		t.Errorf("RandomBase = %+v, want StrategyPct=0.82 Beats=false OK=true", detail.RandomBase)
	}
}

func TestSaveWalkForward_DSRNotOkPersistsZero(t *testing.T) {
	s := openTestStore(t)

	meta := RunMeta{Strategy: "rsi2", Horizon: "short-term", Symbol: "SPY", DataPath: "data/SPY.csv"}
	res := fakeWalkForwardResult()
	res.DSR = 0.99 // should be ignored/zeroed because DSROk is false
	res.DSROk = false

	id, err := s.SaveWalkForward(meta, res)
	if err != nil {
		t.Fatalf("SaveWalkForward() error = %v", err)
	}

	detail, err := s.GetRun(id)
	if err != nil {
		t.Fatalf("GetRun() error = %v", err)
	}
	if detail.DSR != 0 {
		t.Errorf("DSR = %v, want 0 when DSROk is false", detail.DSR)
	}
}

func TestSaveKFold_RoundTrip(t *testing.T) {
	s := openTestStore(t)

	meta := RunMeta{
		Strategy: "rsi2",
		Horizon:  "short-term",
		Symbol:   "SPY",
		DataPath: "data/SPY.csv",
		Config:   Config{Cash: 100000, SlippageBps: 5, Folds: 4, PurgeBars: 20, EmbargoBars: 6},
	}
	res := fakeKFoldResult()

	id, err := s.SaveKFold(meta, res)
	if err != nil {
		t.Fatalf("SaveKFold() error = %v", err)
	}

	detail, err := s.GetRun(id)
	if err != nil {
		t.Fatalf("GetRun() error = %v", err)
	}

	if detail.Kind != "kfold" {
		t.Errorf("Kind = %q, want kfold", detail.Kind)
	}
	if detail.Trials != 36 {
		t.Errorf("Trials = %d, want 36 (TotalTrials)", detail.Trials)
	}
	if detail.Fills != 10 {
		t.Errorf("Fills = %d, want 10 (TotalFills)", detail.Fills)
	}
	if detail.Metrics.Sharpe != 1.2 || detail.Metrics.CAGR != 0.15 {
		t.Errorf("Metrics = %+v, want OOS Sharpe=1.2 CAGR=0.15", detail.Metrics)
	}
	if detail.TrialSRVar != 0.02 {
		t.Errorf("TrialSRVar = %v, want 0.02", detail.TrialSRVar)
	}
	if detail.DSR != 0.87 {
		t.Errorf("DSR = %v, want 0.87", detail.DSR)
	}

	if len(detail.Folds) != 2 {
		t.Fatalf("len(Folds) = %d, want 2", len(detail.Folds))
	}

	f0 := detail.Folds[0]
	if f0.Fold != 1 {
		t.Errorf("Folds[0].Fold = %d, want 1", f0.Fold)
	}
	if f0.TrainStart != "" || f0.TrainEnd != "" {
		t.Errorf("Folds[0].TrainStart/End = %q/%q, want empty strings for kfold", f0.TrainStart, f0.TrainEnd)
	}
	if f0.TestStart != "2015-01-01" || f0.TestEnd != "2015-12-31" {
		t.Errorf("Folds[0] TestStart/End = %q/%q, want 2015-01-01/2015-12-31", f0.TestStart, f0.TestEnd)
	}
	if f0.BestParams["entryRSI"] != 5 || f0.BestParams["exitRSI"] != 60 {
		t.Errorf("Folds[0].BestParams = %+v, want entryRSI=5 exitRSI=60", f0.BestParams)
	}
	if f0.Fills != 4 || f0.Trials != 18 {
		t.Errorf("Folds[0].Fills/Trials = %d/%d, want 4/18", f0.Fills, f0.Trials)
	}

	f1 := detail.Folds[1]
	if f1.TrainStart != "" || f1.TrainEnd != "" {
		t.Errorf("Folds[1].TrainStart/End = %q/%q, want empty strings for kfold", f1.TrainStart, f1.TrainEnd)
	}

	// Config JSON round-trips the kfold-only fields.
	if detail.Config.Folds != 4 || detail.Config.PurgeBars != 20 || detail.Config.EmbargoBars != 6 {
		t.Errorf("Config = %+v, want Folds=4 PurgeBars=20 EmbargoBars=6", detail.Config)
	}
	if detail.Config.Cash != 100000 || detail.Config.SlippageBps != 5 {
		t.Errorf("Config = %+v, want Cash=100000 SlippageBps=5", detail.Config)
	}

	if len(detail.Equity) != 3 {
		t.Fatalf("len(Equity) = %d, want 3", len(detail.Equity))
	}
	if detail.Benchmark == nil {
		t.Fatal("Benchmark = nil, want non-nil for a kfold run")
	}
	if detail.Benchmark.Sharpe != 0.6 {
		t.Errorf("Benchmark.Sharpe = %v, want 0.6", detail.Benchmark.Sharpe)
	}

	if detail.Regimes == nil {
		t.Fatal("Regimes = nil, want non-nil for a kfold run")
	}
	if len(detail.Regimes.Slices) != 3 {
		t.Fatalf("len(Regimes.Slices) = %d, want 3", len(detail.Regimes.Slices))
	}
	if detail.Regimes.Dropped != 2 {
		t.Errorf("Regimes.Dropped = %d, want 2", detail.Regimes.Dropped)
	}

	if detail.RandomBase == nil {
		t.Fatal("RandomBase = nil, want non-nil for a kfold run")
	}
	if detail.RandomBase.Blocks != 12 || detail.RandomBase.BlockLen != 15 {
		t.Errorf("RandomBase = %+v, want Blocks=12 BlockLen=15", detail.RandomBase)
	}
}

func TestSaveKFold_DSRNotOkPersistsZero(t *testing.T) {
	s := openTestStore(t)

	meta := RunMeta{Strategy: "rsi2", Horizon: "short-term", Symbol: "SPY", DataPath: "data/SPY.csv"}
	res := fakeKFoldResult()
	res.DSR = 0.99 // should be ignored/zeroed because DSROk is false
	res.DSROk = false

	id, err := s.SaveKFold(meta, res)
	if err != nil {
		t.Fatalf("SaveKFold() error = %v", err)
	}

	detail, err := s.GetRun(id)
	if err != nil {
		t.Fatalf("GetRun() error = %v", err)
	}
	if detail.DSR != 0 {
		t.Errorf("DSR = %v, want 0 when DSROk is false", detail.DSR)
	}
}

func TestOpen_MigrationIsIdempotent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "test.db")

	s1, err := Open(path)
	if err != nil {
		t.Fatalf("Open() [1st] error = %v", err)
	}
	if err := s1.Close(); err != nil {
		t.Fatalf("Close() [1st] error = %v", err)
	}

	// Re-opening an already-migrated database must not fail on "duplicate
	// column" from re-running ADD COLUMN.
	s2, err := Open(path)
	if err != nil {
		t.Fatalf("Open() [2nd] error = %v", err)
	}
	defer s2.Close()

	meta := RunMeta{Strategy: "sma-cross", Horizon: "long-term", Symbol: "SPY", DataPath: "data/SPY.csv"}
	if _, err := s2.SaveBacktest(meta, fakeBacktestResult()); err != nil {
		t.Fatalf("SaveBacktest() after re-open error = %v", err)
	}
}

func TestListRuns_IncludesDSRAndTrialSRVar(t *testing.T) {
	s := openTestStore(t)

	meta := RunMeta{Strategy: "rsi2", Horizon: "short-term", Symbol: "SPY", DataPath: "data/SPY.csv"}
	if _, err := s.SaveWalkForward(meta, fakeWalkForwardResult()); err != nil {
		t.Fatalf("SaveWalkForward() error = %v", err)
	}

	runs, err := s.ListRuns(0)
	if err != nil {
		t.Fatalf("ListRuns() error = %v", err)
	}
	if len(runs) != 1 {
		t.Fatalf("len(runs) = %d, want 1", len(runs))
	}
	if runs[0].TrialSRVar != 0.02 {
		t.Errorf("TrialSRVar = %v, want 0.02", runs[0].TrialSRVar)
	}
	if runs[0].DSR != 0.87 {
		t.Errorf("DSR = %v, want 0.87", runs[0].DSR)
	}
}

func TestListRuns_OrderingAndLimit(t *testing.T) {
	s := openTestStore(t)

	meta := RunMeta{Strategy: "sma-cross", Horizon: "long-term", Symbol: "SPY", DataPath: "data/SPY.csv"}

	id1, err := s.SaveBacktest(meta, fakeBacktestResult())
	if err != nil {
		t.Fatalf("SaveBacktest() #1 error = %v", err)
	}
	id2, err := s.SaveBacktest(meta, fakeBacktestResult())
	if err != nil {
		t.Fatalf("SaveBacktest() #2 error = %v", err)
	}
	if id2 <= id1 {
		t.Fatalf("expected id2 (%d) > id1 (%d)", id2, id1)
	}

	all, err := s.ListRuns(0)
	if err != nil {
		t.Fatalf("ListRuns(0) error = %v", err)
	}
	if len(all) != 2 {
		t.Fatalf("ListRuns(0) len = %d, want 2", len(all))
	}
	if all[0].Id != id2 || all[1].Id != id1 {
		t.Errorf("ListRuns(0) order = [%d, %d], want newest first [%d, %d]", all[0].Id, all[1].Id, id2, id1)
	}

	limited, err := s.ListRuns(1)
	if err != nil {
		t.Fatalf("ListRuns(1) error = %v", err)
	}
	if len(limited) != 1 {
		t.Fatalf("ListRuns(1) len = %d, want 1", len(limited))
	}
	if limited[0].Id != id2 {
		t.Errorf("ListRuns(1)[0].Id = %d, want newest %d", limited[0].Id, id2)
	}

	if limited[0].Sharpe != 1.23 {
		t.Errorf("ListRuns()[0].Sharpe = %v, want 1.23", limited[0].Sharpe)
	}
	if limited[0].MaxDrawdown != 0.05 {
		t.Errorf("ListRuns()[0].MaxDrawdown = %v, want 0.05", limited[0].MaxDrawdown)
	}
}

func TestGetRun_MissingIdErrors(t *testing.T) {
	s := openTestStore(t)

	if _, err := s.GetRun(999); err == nil {
		t.Fatal("GetRun(999) error = nil, want error for missing id")
	}
}

// TestGetRun_LegacyRowWithoutEquityOrBenchmark simulates a row saved before
// the equity_json/benchmark_json/regime_json/randbase_json columns existed:
// addColumnIfMissing backs such rows with ” (see Open's migration), which
// GetRun must treat as "not persisted" (nil), not as an unmarshal error.
func TestGetRun_LegacyRowWithoutEquityOrBenchmark(t *testing.T) {
	s := openTestStore(t)

	_, err := s.db.Exec(
		`INSERT INTO runs (created_at, kind, strategy, horizon, symbol, data_path,
			period_start, period_end, config_json, metrics_json, trials, fills,
			trial_sr_var, dsr, equity_json, benchmark_json)
		 VALUES (?, 'backtest', 'sma-cross', 'long-term', 'SPY', 'data/SPY.csv',
			'2020-01-01', '2020-12-31', '{}', '{}', 0, 2, 0, 0, '', '')`,
		time.Now().UTC().Format(time.RFC3339),
	)
	if err != nil {
		t.Fatalf("insert legacy row: %v", err)
	}

	detail, err := s.GetRun(1)
	if err != nil {
		t.Fatalf("GetRun() error = %v", err)
	}
	if detail.Equity != nil {
		t.Errorf("Equity = %+v, want nil for a legacy row", detail.Equity)
	}
	if detail.Benchmark != nil {
		t.Errorf("Benchmark = %+v, want nil for a legacy row", detail.Benchmark)
	}
	if detail.Regimes != nil {
		t.Errorf("Regimes = %+v, want nil for a legacy row", detail.Regimes)
	}
	if detail.RandomBase != nil {
		t.Errorf("RandomBase = %+v, want nil for a legacy row", detail.RandomBase)
	}
}
