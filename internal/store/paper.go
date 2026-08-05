package store

import (
	"database/sql"
	"fmt"
	"time"
)

// PaperRun is one persisted paper-trading session (internal/paper.Session.
// RunOnce), recorded whether it ran dry-run or execute, and regardless of
// whether any order was actually approved.
type PaperRun struct {
	Id           int64
	Ts           time.Time
	Mode         string // "dry-run" | "execute"
	Account      string
	EquityMicros int64 // domain.Money in micros — cardinal rule 6
	CashMicros   int64 // domain.Money in micros — cardinal rule 6
	Detail       string
	// TraceJSON is a Traced strategy's deliberation trace for this session
	// (agent-committee's CommitteeTrace, verbatim JSON), "" when no traced
	// strategy ran or the row predates the trace_json column.
	TraceJSON string
	// AdvisoryJSON is the LLM advisory attached to this session
	// (internal/agent.Advisory, verbatim JSON), "" when the advisor was
	// disabled or the row predates the advisory_json column.
	AdvisoryJSON string
}

// PaperOrder is one order (approved, dry-run, rejected, or executed) recorded
// as part of a PaperRun.
type PaperOrder struct {
	Id             int64
	RunID          int64
	Ts             time.Time
	Strategy       string
	Symbol         string
	Side           string
	Qty            int64
	RefPriceMicros int64 // domain.Money in micros — cardinal rule 6
	Status         string
	OrderID        string
	Detail         string
}

// SavePaperRun persists one paper-trading session and returns its new row
// id. Callers pass ts explicitly (rather than the store stamping time.Now())
// so internal/paper.Session's NowFunc field keeps every timestamp in a
// session's ledger consistent and testable.
func (s *Store) SavePaperRun(ts time.Time, mode, account string, equityMicros, cashMicros int64, detailJSON string) (int64, error) {
	result, err := s.db.Exec(
		`INSERT INTO paper_runs (ts, mode, account, equity_micros, cash_micros, detail_json)
		 VALUES (?, ?, ?, ?, ?, ?)`,
		ts.UTC().Format(time.RFC3339),
		mode,
		account,
		equityMicros,
		cashMicros,
		detailJSON,
	)
	if err != nil {
		return 0, fmt.Errorf("store: insert paper run: %w", err)
	}
	id, err := result.LastInsertId()
	if err != nil {
		return 0, fmt.Errorf("store: last insert id: %w", err)
	}
	return id, nil
}

// SetPaperRunAgent attaches the agentic layer's per-session artifacts to an
// already-saved paper run: a Traced strategy's deliberation trace and/or the
// LLM advisory (either may be ""). A separate additive method — rather than
// widening SavePaperRun's signature — so every existing SavePaperRun call
// site (and the ledger rows already on disk) is untouched by the agentic
// layer landing.
func (s *Store) SetPaperRunAgent(runID int64, traceJSON, advisoryJSON string) error {
	if _, err := s.db.Exec(
		`UPDATE paper_runs SET trace_json = ?, advisory_json = ? WHERE id = ?`,
		traceJSON, advisoryJSON, runID,
	); err != nil {
		return fmt.Errorf("store: set paper run %d agent artifacts: %w", runID, err)
	}
	return nil
}

// UpdatePaperOrderStatus overwrites one paper order's status by row id. Used
// by the paper session's start-of-run reconciliation: an order placed after
// hours is recorded as "PreSubmitted"/"Submitted" and fills at the next open
// AFTER the session process has exited, so the ledger must be caught up
// against the gateway before position ages are derived from it.
func (s *Store) UpdatePaperOrderStatus(rowID int64, status string) error {
	if _, err := s.db.Exec(`UPDATE paper_orders SET status = ? WHERE id = ?`, status, rowID); err != nil {
		return fmt.Errorf("store: update paper order %d status: %w", rowID, err)
	}
	return nil
}

// SavePaperOrder persists one order belonging to runID and returns its new
// row id.
func (s *Store) SavePaperOrder(runID int64, ts time.Time, strategy, symbol, side string, qty, refPriceMicros int64, status, orderID, detail string) (int64, error) {
	result, err := s.db.Exec(
		`INSERT INTO paper_orders (run_id, ts, strategy, symbol, side, qty, ref_price_micros, status, order_id, detail)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		runID,
		ts.UTC().Format(time.RFC3339),
		strategy,
		symbol,
		side,
		qty,
		refPriceMicros,
		status,
		orderID,
		detail,
	)
	if err != nil {
		return 0, fmt.Errorf("store: insert paper order: %w", err)
	}
	id, err := result.LastInsertId()
	if err != nil {
		return 0, fmt.Errorf("store: last insert id: %w", err)
	}
	return id, nil
}

// ListPaperRuns returns up to limit paper-trading sessions, newest first.
// limit <= 0 means no limit.
func (s *Store) ListPaperRuns(limit int) ([]PaperRun, error) {
	query := `SELECT id, ts, mode, account, equity_micros, cash_micros, detail_json, trace_json, advisory_json
		FROM paper_runs ORDER BY id DESC`
	args := []any{}
	if limit > 0 {
		query += ` LIMIT ?`
		args = append(args, limit)
	}

	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, fmt.Errorf("store: list paper runs: %w", err)
	}
	defer rows.Close()

	var out []PaperRun
	for rows.Next() {
		var (
			r  PaperRun
			ts string
		)
		if err := rows.Scan(&r.Id, &ts, &r.Mode, &r.Account, &r.EquityMicros, &r.CashMicros, &r.Detail, &r.TraceJSON, &r.AdvisoryJSON); err != nil {
			return nil, fmt.Errorf("store: scan paper run: %w", err)
		}
		parsed, err := time.Parse(time.RFC3339, ts)
		if err != nil {
			return nil, fmt.Errorf("store: parse ts %q: %w", ts, err)
		}
		r.Ts = parsed
		out = append(out, r)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("store: list paper runs: %w", err)
	}
	return out, nil
}

// GetPaperRun returns the paper-trading session with the given row id, or
// (zero, false) when no such row exists. Callers that know exactly which run
// they mean (e.g. the dry-run handler attaching the trace/advisory of the
// session it just ran) must use this instead of assuming the newest
// ListPaperRuns row is theirs — a concurrent session can interleave.
func (s *Store) GetPaperRun(id int64) (PaperRun, bool, error) {
	var (
		r  PaperRun
		ts string
	)
	err := s.db.QueryRow(
		`SELECT id, ts, mode, account, equity_micros, cash_micros, detail_json, trace_json, advisory_json
		 FROM paper_runs WHERE id = ?`, id).
		Scan(&r.Id, &ts, &r.Mode, &r.Account, &r.EquityMicros, &r.CashMicros, &r.Detail, &r.TraceJSON, &r.AdvisoryJSON)
	if err == sql.ErrNoRows {
		return PaperRun{}, false, nil
	}
	if err != nil {
		return PaperRun{}, false, fmt.Errorf("store: get paper run %d: %w", id, err)
	}
	parsed, err := time.Parse(time.RFC3339, ts)
	if err != nil {
		return PaperRun{}, false, fmt.Errorf("store: parse ts %q: %w", ts, err)
	}
	r.Ts = parsed
	return r, true, nil
}

// ListPaperOrders returns every order recorded for runID, oldest first.
func (s *Store) ListPaperOrders(runID int64) ([]PaperOrder, error) {
	rows, err := s.db.Query(
		`SELECT id, run_id, ts, strategy, symbol, side, qty, ref_price_micros, status, order_id, detail
		 FROM paper_orders WHERE run_id = ? ORDER BY id ASC`, runID)
	if err != nil {
		return nil, fmt.Errorf("store: list paper orders for run %d: %w", runID, err)
	}
	defer rows.Close()

	var out []PaperOrder
	for rows.Next() {
		var (
			o  PaperOrder
			ts string
		)
		if err := rows.Scan(&o.Id, &o.RunID, &ts, &o.Strategy, &o.Symbol, &o.Side, &o.Qty,
			&o.RefPriceMicros, &o.Status, &o.OrderID, &o.Detail); err != nil {
			return nil, fmt.Errorf("store: scan paper order: %w", err)
		}
		parsed, err := time.Parse(time.RFC3339, ts)
		if err != nil {
			return nil, fmt.Errorf("store: parse ts %q: %w", ts, err)
		}
		o.Ts = parsed
		out = append(out, o)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("store: list paper orders for run %d: %w", runID, err)
	}
	return out, nil
}

// LatestPaperEquityPeak returns the highest equity_micros ever recorded
// across all paper_runs, or (0, false) if no paper run has been saved yet.
// internal/paper.Session uses this to seed the risk manager's drawdown
// kill-switch peak across sessions (a single RunOnce call has no in-memory
// history of its own — each invocation is a fresh process).
func (s *Store) LatestPaperEquityPeak() (int64, bool, error) {
	var peak sql.NullInt64
	row := s.db.QueryRow(`SELECT MAX(equity_micros) FROM paper_runs`)
	if err := row.Scan(&peak); err != nil {
		return 0, false, fmt.Errorf("store: latest paper equity peak: %w", err)
	}
	if !peak.Valid {
		return 0, false, nil
	}
	return peak.Int64, true, nil
}
