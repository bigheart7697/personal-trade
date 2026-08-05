// Package store persists backtest and walk-forward run results to a local
// SQLite database so results accumulate across sessions and are comparable
// (list, filter, inspect by id) instead of vanishing with the terminal that
// produced them. Persistence here is deliberately best-effort from the
// CLI's point of view: a strategy's backtest or walk-forward result is the
// product; a store failure must never fail the run that produced it (see
// cmd/tradeforge/main.go's save-after-run wiring).
//
// This is the project's first dependency beyond the standard library:
// modernc.org/sqlite is a pure-Go SQLite driver (no cgo), so it costs
// nothing extra on the Windows dev machine (no gcc toolchain). See
// docs/OPTIONS.md §Dependencies for the justification.
package store

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	_ "modernc.org/sqlite" // registers the "sqlite" database/sql driver

	"tradeforge/internal/backtest"
	"tradeforge/internal/eval"
	"tradeforge/internal/metrics"
)

const schema = `
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  kind TEXT NOT NULL,
  strategy TEXT NOT NULL,
  horizon TEXT NOT NULL,
  symbol TEXT NOT NULL,
  data_path TEXT NOT NULL,
  period_start TEXT NOT NULL,
  period_end TEXT NOT NULL,
  config_json TEXT NOT NULL,
  metrics_json TEXT NOT NULL,
  trials INTEGER NOT NULL DEFAULT 0,
  fills INTEGER NOT NULL DEFAULT 0
);
-- train_start/train_end are empty strings for kfold folds: a kfold fold
-- trains on (up to two) disjoint segments both sides of the test window, so
-- a single train date range is meaningless. Empty strings mark that case;
-- walkforward folds always populate both.
CREATE TABLE IF NOT EXISTS folds (
  run_id INTEGER NOT NULL REFERENCES runs(id),
  fold INTEGER NOT NULL,
  train_start TEXT NOT NULL, train_end TEXT NOT NULL,
  test_start TEXT NOT NULL,  test_end TEXT NOT NULL,
  best_params_json TEXT NOT NULL,
  train_objective REAL NOT NULL,
  test_metrics_json TEXT NOT NULL,
  fills INTEGER NOT NULL,
  trials INTEGER NOT NULL,
  PRIMARY KEY (run_id, fold)
);
-- paper_runs/paper_orders are the paper-trading session ledger (internal/paper,
-- docs/OPTIONS.md §6a's daily-session design). Cardinal rule 6 (fixed-point
-- accounting) applies to this ledger: equity_micros/cash_micros/ref_price_micros
-- are domain.Money (int64 micros), never float64, even though the rest of this
-- store (backtest/walkforward/kfold metrics) legitimately stays float64 per the
-- research-vs-broker-facing boundary documented in internal/domain/money.go.
CREATE TABLE IF NOT EXISTS paper_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  mode TEXT NOT NULL,
  account TEXT NOT NULL,
  equity_micros INTEGER NOT NULL,
  cash_micros INTEGER NOT NULL,
  detail_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_orders (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL REFERENCES paper_runs(id),
  ts TEXT NOT NULL,
  strategy TEXT NOT NULL,
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,
  qty INTEGER NOT NULL,
  ref_price_micros INTEGER NOT NULL,
  status TEXT NOT NULL,
  order_id TEXT NOT NULL,
  detail TEXT NOT NULL
);
`

// Store is a handle to the run-persistence database. It is safe for
// concurrent use (database/sql pools connections internally).
type Store struct {
	db *sql.DB
}

// Config carries the kind-agnostic fields describing a run, independent of
// whether it was a plain backtest or a walk-forward evaluation.
type Config struct {
	Cash               float64 `json:"cash"`
	SlippageBps        float64 `json:"slippageBps"`
	CommissionPerShare float64 `json:"commissionPerShare,omitempty"`
	MinCommission      float64 `json:"minCommission,omitempty"`
	TrainBars          int     `json:"trainBars,omitempty"`
	TestBars           int     `json:"testBars,omitempty"`

	// Folds/PurgeBars/EmbargoBars are kfold-only. Added after TrainBars/
	// TestBars; omitempty keeps old rows (which lack these keys) parsing
	// back with zero values, backward-compatible.
	Folds       int `json:"folds,omitempty"`
	PurgeBars   int `json:"purgeBars,omitempty"`
	EmbargoBars int `json:"embargoBars,omitempty"`

	// BenchmarkSymbol is the resolved multi-symbol benchmark (e.g. "SPY" or
	// "QQQ"), empty for a single-series run. omitempty keeps old rows (saved
	// before this field existed, and every single-series run) parsing back
	// as "", backward-compatible.
	BenchmarkSymbol string `json:"benchmarkSymbol,omitempty"`
}

// RunMeta carries the fields SaveBacktest/SaveWalkForward need beyond what
// backtest.Result / eval.Result already carry (data provenance and run
// configuration, which the engine results don't track).
type RunMeta struct {
	Strategy string
	Horizon  string
	Symbol   string
	DataPath string
	Config   Config
}

// RunSummary is one row of ListRuns output: enough to render a table
// without unmarshaling metrics_json for every caller.
type RunSummary struct {
	Id          int64
	CreatedAt   time.Time
	Kind        string
	Strategy    string
	Horizon     string
	Symbol      string
	PeriodStart string
	PeriodEnd   string
	Sharpe      float64
	MaxDrawdown float64
	CAGR        float64
	Trials      int
	Fills       int
	TrialSRVar  float64
	DSR         float64
}

// FoldDetail is one unmarshaled folds row, returned as part of GetRun.
type FoldDetail struct {
	Fold           int
	TrainStart     string
	TrainEnd       string
	TestStart      string
	TestEnd        string
	BestParams     map[string]float64
	TrainObjective float64
	TestMetrics    metrics.Metrics
	Fills          int
	Trials         int
}

// EquityPointJSON is one point of a persisted equity curve: a compact
// {"t":"2006-01-02","v":100000.0} shape shared with the dashboard's existing
// /api/backtest response (internal/server).
type EquityPointJSON struct {
	T string  `json:"t"`
	V float64 `json:"v"`
}

// RegimeReport folds eval.RegimeBreakdown's two return values (the fixed
// three-slice breakdown and the dropped-day count) into a single struct so
// it round-trips through one JSON column (regime_json) as one JSON object.
type RegimeReport struct {
	Slices  []eval.RegimeSlice `json:"slices"`
	Dropped int                `json:"dropped"`
}

// RunDetail is the full unmarshaled runs row plus its folds (empty for a
// plain backtest), as returned by GetRun.
type RunDetail struct {
	Id          int64
	CreatedAt   time.Time
	Kind        string
	Strategy    string
	Horizon     string
	Symbol      string
	DataPath    string
	PeriodStart string
	PeriodEnd   string
	Config      Config
	Metrics     metrics.Metrics
	Trials      int
	Fills       int
	TrialSRVar  float64
	DSR         float64
	Folds       []FoldDetail
	// Equity is the run's equity curve (plain backtest: the full curve;
	// walkforward/kfold: the stitched OOS curve), nil when equity_json was
	// never persisted ('' — a legacy row saved before this column existed).
	Equity []EquityPointJSON
	// Benchmark is the buy-and-hold comparison computed alongside a
	// walkforward/kfold run, nil for a plain backtest or a legacy row.
	Benchmark *metrics.Metrics
	// Regimes is the regime breakdown computed alongside a walkforward/kfold
	// run, nil for a plain backtest or a legacy row saved before this column
	// existed.
	Regimes *RegimeReport
	// RandomBase is the random-entry/same-risk baseline computed alongside a
	// walkforward/kfold run, nil for a plain backtest or a legacy row saved
	// before this column existed.
	RandomBase *eval.RandomBaseline
}

// Open opens (creating if necessary) the SQLite database at path, ensuring
// its parent directory exists, and creates the schema if it is not already
// present.
func Open(path string) (*Store, error) {
	// A path that is itself a directory would otherwise reach the SQLite
	// driver and surface as the driver's misleading "out of memory (14)"
	// (result code 14 is SQLITE_CANTOPEN, mismapped) — catch the common
	// misuse up front with a clear message.
	if info, err := os.Stat(path); err == nil && info.IsDir() {
		return nil, fmt.Errorf("store: %s is a directory, not a database file", path)
	}

	dir := filepath.Dir(path)
	if dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, fmt.Errorf("store: create directory %s: %w", dir, err)
		}
	}

	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("store: open %s: %w", path, err)
	}

	if err := db.Ping(); err != nil {
		db.Close()
		return nil, fmt.Errorf("store: ping %s: %w", path, err)
	}

	if _, err := db.Exec(schema); err != nil {
		db.Close()
		return nil, fmt.Errorf("store: create schema: %w", err)
	}

	// Runs 1-6 (per STATE.md) were created before the deflated-Sharpe columns
	// existed. CREATE TABLE IF NOT EXISTS above is a no-op against an
	// already-existing runs table, so those columns must be added out of
	// band via ALTER TABLE. SQLite has no "ADD COLUMN IF NOT EXISTS", so the
	// helper below runs the ALTER unconditionally and swallows only the
	// "duplicate column" error a second (or Nth) Open against the same
	// up-to-date database would otherwise return.
	if err := addColumnIfMissing(db, "runs", "trial_sr_var", "REAL NOT NULL DEFAULT 0"); err != nil {
		db.Close()
		return nil, fmt.Errorf("store: migrate schema: %w", err)
	}
	if err := addColumnIfMissing(db, "runs", "dsr", "REAL NOT NULL DEFAULT 0"); err != nil {
		db.Close()
		return nil, fmt.Errorf("store: migrate schema: %w", err)
	}
	// equity_json/benchmark_json were added once the dashboard grew a run
	// detail view (see internal/server): runs saved before this column
	// existed have '' in both, which GetRun treats as "no curve/benchmark
	// persisted" rather than an error.
	if err := addColumnIfMissing(db, "runs", "equity_json", "TEXT NOT NULL DEFAULT ''"); err != nil {
		db.Close()
		return nil, fmt.Errorf("store: migrate schema: %w", err)
	}
	if err := addColumnIfMissing(db, "runs", "benchmark_json", "TEXT NOT NULL DEFAULT ''"); err != nil {
		db.Close()
		return nil, fmt.Errorf("store: migrate schema: %w", err)
	}
	// regime_json/randbase_json were added once the evaluation report grew a
	// regime breakdown and a random-entry baseline (docs/ROADMAP.md Phase
	// 2): runs saved before this column existed (and every plain backtest,
	// which has neither) have '' in both, which GetRun treats as "not
	// computed for this run" (nil) rather than an error.
	if err := addColumnIfMissing(db, "runs", "regime_json", "TEXT NOT NULL DEFAULT ''"); err != nil {
		db.Close()
		return nil, fmt.Errorf("store: migrate schema: %w", err)
	}
	if err := addColumnIfMissing(db, "runs", "randbase_json", "TEXT NOT NULL DEFAULT ''"); err != nil {
		db.Close()
		return nil, fmt.Errorf("store: migrate schema: %w", err)
	}
	// trace_json/advisory_json were added with the agentic layer
	// (agent-committee's deliberation trace + the optional LLM advisory —
	// internal/agent, docs/AGENT.md): paper runs saved before these columns
	// existed have '' in both, which readers treat as "not recorded for this
	// session" rather than an error.
	if err := addColumnIfMissing(db, "paper_runs", "trace_json", "TEXT NOT NULL DEFAULT ''"); err != nil {
		db.Close()
		return nil, fmt.Errorf("store: migrate schema: %w", err)
	}
	if err := addColumnIfMissing(db, "paper_runs", "advisory_json", "TEXT NOT NULL DEFAULT ''"); err != nil {
		db.Close()
		return nil, fmt.Errorf("store: migrate schema: %w", err)
	}

	return &Store{db: db}, nil
}

// addColumnIfMissing runs "ALTER TABLE <table> ADD COLUMN <column> <def>",
// swallowing only the driver's "duplicate column" error (matched
// case-insensitively, since the exact wording is driver-specific) so that
// re-running the migration against an already-migrated database is a
// harmless no-op. Any other error propagates.
func addColumnIfMissing(db *sql.DB, table, column, def string) error {
	_, err := db.Exec(fmt.Sprintf("ALTER TABLE %s ADD COLUMN %s %s", table, column, def))
	if err == nil {
		return nil
	}
	if strings.Contains(strings.ToLower(err.Error()), "duplicate column") {
		return nil
	}
	return fmt.Errorf("add column %s.%s: %w", table, column, err)
}

// marshalEquity converts an engine/eval equity curve into the compact JSON
// shape the dashboard consumes, matching /api/backtest's existing point
// shape (internal/server).
func marshalEquity(curve []metrics.EquityPoint) ([]byte, error) {
	points := make([]EquityPointJSON, len(curve))
	for i, pt := range curve {
		points[i] = EquityPointJSON{T: pt.Time.Format("2006-01-02"), V: pt.Equity}
	}
	b, err := json.Marshal(points)
	if err != nil {
		return nil, fmt.Errorf("marshal equity curve: %w", err)
	}
	return b, nil
}

// Close closes the underlying database handle.
func (s *Store) Close() error {
	if err := s.db.Close(); err != nil {
		return fmt.Errorf("store: close: %w", err)
	}
	return nil
}

// SaveBacktest persists a plain backtest run (kind "backtest", trials 0)
// and returns its new row id.
func (s *Store) SaveBacktest(meta RunMeta, res backtest.Result) (int64, error) {
	configJSON, err := json.Marshal(meta.Config)
	if err != nil {
		return 0, fmt.Errorf("store: marshal config: %w", err)
	}
	metricsJSON, err := json.Marshal(res.Metrics)
	if err != nil {
		return 0, fmt.Errorf("store: marshal metrics: %w", err)
	}
	equityJSON, err := marshalEquity(res.EquityCurve)
	if err != nil {
		return 0, fmt.Errorf("store: %w", err)
	}

	result, err := s.db.Exec(
		`INSERT INTO runs (created_at, kind, strategy, horizon, symbol, data_path,
			period_start, period_end, config_json, metrics_json, trials, fills,
			trial_sr_var, dsr, equity_json, benchmark_json, regime_json, randbase_json)
		 VALUES (?, 'backtest', ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 0, 0, ?, '', '', '')`,
		time.Now().UTC().Format(time.RFC3339),
		meta.Strategy,
		meta.Horizon,
		meta.Symbol,
		meta.DataPath,
		res.PeriodStart.Format("2006-01-02"),
		res.PeriodEnd.Format("2006-01-02"),
		string(configJSON),
		string(metricsJSON),
		len(res.Fills),
		string(equityJSON),
	)
	if err != nil {
		return 0, fmt.Errorf("store: insert run: %w", err)
	}

	id, err := result.LastInsertId()
	if err != nil {
		return 0, fmt.Errorf("store: last insert id: %w", err)
	}
	return id, nil
}

// SaveWalkForward persists a walk-forward run (kind "walkforward", metrics
// = stitched OOS metrics, trials/fills = totals) plus one folds row per
// fold, in a single transaction.
func (s *Store) SaveWalkForward(meta RunMeta, res eval.Result) (int64, error) {
	configJSON, err := json.Marshal(meta.Config)
	if err != nil {
		return 0, fmt.Errorf("store: marshal config: %w", err)
	}
	metricsJSON, err := json.Marshal(res.OOSMetrics)
	if err != nil {
		return 0, fmt.Errorf("store: marshal metrics: %w", err)
	}
	equityJSON, err := marshalEquity(res.OOSEquity)
	if err != nil {
		return 0, fmt.Errorf("store: %w", err)
	}
	benchmarkJSON, err := json.Marshal(res.BenchmarkMetrics)
	if err != nil {
		return 0, fmt.Errorf("store: marshal benchmark metrics: %w", err)
	}
	regimeJSON, err := json.Marshal(RegimeReport{Slices: res.Regimes, Dropped: res.RegimeDropped})
	if err != nil {
		return 0, fmt.Errorf("store: marshal regime breakdown: %w", err)
	}
	randbaseJSON, err := json.Marshal(res.RandomBase)
	if err != nil {
		return 0, fmt.Errorf("store: marshal random baseline: %w", err)
	}

	periodStart, periodEnd := "", ""
	if len(res.Folds) > 0 {
		periodStart = res.Folds[0].TrainStart.Format("2006-01-02")
		periodEnd = res.Folds[len(res.Folds)-1].TestEnd.Format("2006-01-02")
	}

	// DSR is only meaningful when it was actually computable; report 0
	// (rather than persisting a degenerate/garbage value) when !DSROk.
	dsr := res.DSR
	if !res.DSROk {
		dsr = 0
	}

	tx, err := s.db.Begin()
	if err != nil {
		return 0, fmt.Errorf("store: begin transaction: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck // no-op if already committed

	result, err := tx.Exec(
		`INSERT INTO runs (created_at, kind, strategy, horizon, symbol, data_path,
			period_start, period_end, config_json, metrics_json, trials, fills,
			trial_sr_var, dsr, equity_json, benchmark_json, regime_json, randbase_json)
		 VALUES (?, 'walkforward', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		time.Now().UTC().Format(time.RFC3339),
		meta.Strategy,
		meta.Horizon,
		meta.Symbol,
		meta.DataPath,
		periodStart,
		periodEnd,
		string(configJSON),
		string(metricsJSON),
		res.TotalTrials,
		res.TotalFills,
		res.TrialSRVar,
		dsr,
		string(equityJSON),
		string(benchmarkJSON),
		string(regimeJSON),
		string(randbaseJSON),
	)
	if err != nil {
		return 0, fmt.Errorf("store: insert run: %w", err)
	}

	runID, err := result.LastInsertId()
	if err != nil {
		return 0, fmt.Errorf("store: last insert id: %w", err)
	}

	for _, fold := range res.Folds {
		paramsJSON := []byte("{}")
		if fold.BestParams != nil {
			paramsJSON, err = json.Marshal(fold.BestParams)
			if err != nil {
				return 0, fmt.Errorf("store: marshal fold %d params: %w", fold.Fold, err)
			}
		}
		testMetricsJSON, err := json.Marshal(fold.TestMetrics)
		if err != nil {
			return 0, fmt.Errorf("store: marshal fold %d metrics: %w", fold.Fold, err)
		}

		if _, err := tx.Exec(
			`INSERT INTO folds (run_id, fold, train_start, train_end, test_start, test_end,
				best_params_json, train_objective, test_metrics_json, fills, trials)
			 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			runID,
			fold.Fold,
			fold.TrainStart.Format("2006-01-02"),
			fold.TrainEnd.Format("2006-01-02"),
			fold.TestStart.Format("2006-01-02"),
			fold.TestEnd.Format("2006-01-02"),
			string(paramsJSON),
			fold.TrainObjective,
			string(testMetricsJSON),
			fold.NumFills,
			fold.Trials,
		); err != nil {
			return 0, fmt.Errorf("store: insert fold %d: %w", fold.Fold, err)
		}
	}

	if err := tx.Commit(); err != nil {
		return 0, fmt.Errorf("store: commit: %w", err)
	}

	return runID, nil
}

// SaveKFold persists a purged & embargoed K-fold run (kind "kfold", metrics
// = stitched OOS metrics, trials/fills = totals) plus one folds row per
// fold, in a single transaction. Reuses the same schema as SaveWalkForward:
// a kfold fold has no single train date range (it trains on disjoint
// segments both sides of the test window), so train_start/train_end are
// persisted as empty strings (see the folds table's schema comment).
// ParamWins is not persisted — it is fully derivable from the folds'
// best_params_json, so storing it separately would just be a denormalized
// duplicate.
func (s *Store) SaveKFold(meta RunMeta, res eval.KFoldResult) (int64, error) {
	configJSON, err := json.Marshal(meta.Config)
	if err != nil {
		return 0, fmt.Errorf("store: marshal config: %w", err)
	}
	metricsJSON, err := json.Marshal(res.OOSMetrics)
	if err != nil {
		return 0, fmt.Errorf("store: marshal metrics: %w", err)
	}
	equityJSON, err := marshalEquity(res.OOSEquity)
	if err != nil {
		return 0, fmt.Errorf("store: %w", err)
	}
	benchmarkJSON, err := json.Marshal(res.BenchmarkMetrics)
	if err != nil {
		return 0, fmt.Errorf("store: marshal benchmark metrics: %w", err)
	}
	regimeJSON, err := json.Marshal(RegimeReport{Slices: res.Regimes, Dropped: res.RegimeDropped})
	if err != nil {
		return 0, fmt.Errorf("store: marshal regime breakdown: %w", err)
	}
	randbaseJSON, err := json.Marshal(res.RandomBase)
	if err != nil {
		return 0, fmt.Errorf("store: marshal random baseline: %w", err)
	}

	periodStart, periodEnd := "", ""
	if len(res.FoldResults) > 0 {
		periodStart = res.FoldResults[0].TestStart.Format("2006-01-02")
		periodEnd = res.FoldResults[len(res.FoldResults)-1].TestEnd.Format("2006-01-02")
	}

	// DSR is only meaningful when it was actually computable; report 0
	// (rather than persisting a degenerate/garbage value) when !DSROk.
	dsr := res.DSR
	if !res.DSROk {
		dsr = 0
	}

	tx, err := s.db.Begin()
	if err != nil {
		return 0, fmt.Errorf("store: begin transaction: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck // no-op if already committed

	result, err := tx.Exec(
		`INSERT INTO runs (created_at, kind, strategy, horizon, symbol, data_path,
			period_start, period_end, config_json, metrics_json, trials, fills,
			trial_sr_var, dsr, equity_json, benchmark_json, regime_json, randbase_json)
		 VALUES (?, 'kfold', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		time.Now().UTC().Format(time.RFC3339),
		meta.Strategy,
		meta.Horizon,
		meta.Symbol,
		meta.DataPath,
		periodStart,
		periodEnd,
		string(configJSON),
		string(metricsJSON),
		res.TotalTrials,
		res.TotalFills,
		res.TrialSRVar,
		dsr,
		string(equityJSON),
		string(benchmarkJSON),
		string(regimeJSON),
		string(randbaseJSON),
	)
	if err != nil {
		return 0, fmt.Errorf("store: insert run: %w", err)
	}

	runID, err := result.LastInsertId()
	if err != nil {
		return 0, fmt.Errorf("store: last insert id: %w", err)
	}

	for _, fold := range res.FoldResults {
		paramsJSON := []byte("{}")
		if fold.BestParams != nil {
			paramsJSON, err = json.Marshal(fold.BestParams)
			if err != nil {
				return 0, fmt.Errorf("store: marshal fold %d params: %w", fold.Fold, err)
			}
		}
		testMetricsJSON, err := json.Marshal(fold.TestMetrics)
		if err != nil {
			return 0, fmt.Errorf("store: marshal fold %d metrics: %w", fold.Fold, err)
		}

		if _, err := tx.Exec(
			`INSERT INTO folds (run_id, fold, train_start, train_end, test_start, test_end,
				best_params_json, train_objective, test_metrics_json, fills, trials)
			 VALUES (?, ?, '', '', ?, ?, ?, ?, ?, ?, ?)`,
			runID,
			fold.Fold,
			fold.TestStart.Format("2006-01-02"),
			fold.TestEnd.Format("2006-01-02"),
			string(paramsJSON),
			fold.TrainObjective,
			string(testMetricsJSON),
			fold.NumFills,
			fold.Trials,
		); err != nil {
			return 0, fmt.Errorf("store: insert fold %d: %w", fold.Fold, err)
		}
	}

	if err := tx.Commit(); err != nil {
		return 0, fmt.Errorf("store: commit: %w", err)
	}

	return runID, nil
}

// ListRuns returns up to limit runs, newest first. limit <= 0 means no
// limit.
func (s *Store) ListRuns(limit int) ([]RunSummary, error) {
	query := `SELECT id, created_at, kind, strategy, horizon, symbol,
		period_start, period_end, metrics_json, trials, fills, trial_sr_var, dsr
		FROM runs ORDER BY id DESC`
	args := []any{}
	if limit > 0 {
		query += ` LIMIT ?`
		args = append(args, limit)
	}

	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, fmt.Errorf("store: list runs: %w", err)
	}
	defer rows.Close()

	var out []RunSummary
	for rows.Next() {
		var (
			sum         RunSummary
			createdAt   string
			metricsJSON string
		)
		if err := rows.Scan(&sum.Id, &createdAt, &sum.Kind, &sum.Strategy, &sum.Horizon,
			&sum.Symbol, &sum.PeriodStart, &sum.PeriodEnd, &metricsJSON, &sum.Trials, &sum.Fills,
			&sum.TrialSRVar, &sum.DSR); err != nil {
			return nil, fmt.Errorf("store: scan run: %w", err)
		}

		t, err := time.Parse(time.RFC3339, createdAt)
		if err != nil {
			return nil, fmt.Errorf("store: parse created_at %q: %w", createdAt, err)
		}
		sum.CreatedAt = t

		var m metrics.Metrics
		if err := json.Unmarshal([]byte(metricsJSON), &m); err != nil {
			return nil, fmt.Errorf("store: unmarshal metrics for run %d: %w", sum.Id, err)
		}
		sum.Sharpe = m.Sharpe
		sum.MaxDrawdown = m.MaxDrawdown
		sum.CAGR = m.CAGR

		out = append(out, sum)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("store: list runs: %w", err)
	}

	return out, nil
}

// GetRun returns the full detail for a single run, including its folds (if
// any). It errors if id does not exist.
func (s *Store) GetRun(id int64) (RunDetail, error) {
	var (
		detail        RunDetail
		createdAt     string
		configJSON    string
		metricsJSON   string
		equityJSON    string
		benchmarkJSON string
		regimeJSON    string
		randbaseJSON  string
	)

	row := s.db.QueryRow(
		`SELECT id, created_at, kind, strategy, horizon, symbol, data_path,
			period_start, period_end, config_json, metrics_json, trials, fills,
			trial_sr_var, dsr, equity_json, benchmark_json, regime_json, randbase_json
		 FROM runs WHERE id = ?`, id)
	if err := row.Scan(&detail.Id, &createdAt, &detail.Kind, &detail.Strategy, &detail.Horizon,
		&detail.Symbol, &detail.DataPath, &detail.PeriodStart, &detail.PeriodEnd,
		&configJSON, &metricsJSON, &detail.Trials, &detail.Fills,
		&detail.TrialSRVar, &detail.DSR, &equityJSON, &benchmarkJSON, &regimeJSON, &randbaseJSON); err != nil {
		if err == sql.ErrNoRows {
			return RunDetail{}, fmt.Errorf("store: run %d not found", id)
		}
		return RunDetail{}, fmt.Errorf("store: get run %d: %w", id, err)
	}

	t, err := time.Parse(time.RFC3339, createdAt)
	if err != nil {
		return RunDetail{}, fmt.Errorf("store: parse created_at %q: %w", createdAt, err)
	}
	detail.CreatedAt = t

	if err := json.Unmarshal([]byte(configJSON), &detail.Config); err != nil {
		return RunDetail{}, fmt.Errorf("store: unmarshal config for run %d: %w", id, err)
	}
	if err := json.Unmarshal([]byte(metricsJSON), &detail.Metrics); err != nil {
		return RunDetail{}, fmt.Errorf("store: unmarshal metrics for run %d: %w", id, err)
	}
	if equityJSON != "" {
		if err := json.Unmarshal([]byte(equityJSON), &detail.Equity); err != nil {
			return RunDetail{}, fmt.Errorf("store: unmarshal equity for run %d: %w", id, err)
		}
	}
	if benchmarkJSON != "" {
		var b metrics.Metrics
		if err := json.Unmarshal([]byte(benchmarkJSON), &b); err != nil {
			return RunDetail{}, fmt.Errorf("store: unmarshal benchmark for run %d: %w", id, err)
		}
		detail.Benchmark = &b
	}
	if regimeJSON != "" {
		var r RegimeReport
		if err := json.Unmarshal([]byte(regimeJSON), &r); err != nil {
			return RunDetail{}, fmt.Errorf("store: unmarshal regime breakdown for run %d: %w", id, err)
		}
		detail.Regimes = &r
	}
	if randbaseJSON != "" {
		var rb eval.RandomBaseline
		if err := json.Unmarshal([]byte(randbaseJSON), &rb); err != nil {
			return RunDetail{}, fmt.Errorf("store: unmarshal random baseline for run %d: %w", id, err)
		}
		detail.RandomBase = &rb
	}

	rows, err := s.db.Query(
		`SELECT fold, train_start, train_end, test_start, test_end,
			best_params_json, train_objective, test_metrics_json, fills, trials
		 FROM folds WHERE run_id = ? ORDER BY fold ASC`, id)
	if err != nil {
		return RunDetail{}, fmt.Errorf("store: list folds for run %d: %w", id, err)
	}
	defer rows.Close()

	for rows.Next() {
		var (
			f               FoldDetail
			paramsJSON      string
			testMetricsJSON string
		)
		if err := rows.Scan(&f.Fold, &f.TrainStart, &f.TrainEnd, &f.TestStart, &f.TestEnd,
			&paramsJSON, &f.TrainObjective, &testMetricsJSON, &f.Fills, &f.Trials); err != nil {
			return RunDetail{}, fmt.Errorf("store: scan fold for run %d: %w", id, err)
		}
		if err := json.Unmarshal([]byte(paramsJSON), &f.BestParams); err != nil {
			return RunDetail{}, fmt.Errorf("store: unmarshal fold %d params for run %d: %w", f.Fold, id, err)
		}
		if err := json.Unmarshal([]byte(testMetricsJSON), &f.TestMetrics); err != nil {
			return RunDetail{}, fmt.Errorf("store: unmarshal fold %d metrics for run %d: %w", f.Fold, id, err)
		}
		detail.Folds = append(detail.Folds, f)
	}
	if err := rows.Err(); err != nil {
		return RunDetail{}, fmt.Errorf("store: list folds for run %d: %w", id, err)
	}

	return detail, nil
}
