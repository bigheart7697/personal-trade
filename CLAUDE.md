# CLAUDE.md — TradeForge

Guidance for working in this repository. Read this before editing.

## What this is

A **personal algorithmic trading platform** for a single user (Reza). It is a *strategy
laboratory first, broker second*: strategies are trained and backtested on historical
data, evaluated under strict validation, promoted to IBKR **paper** trading, and only
ever touch real money after explicit, recorded human approval. Cost-consciousness is a
design goal — the platform exists to make money, so it must not burn money on
infrastructure or API calls.

Sibling prior-art (do not modify): `../Trade`, `../Trade Assistant`, `../Trade pipeline`
are earlier Python experiments. This project is a fresh Go build; mine them for ideas
only if asked.

## Cardinal safety rules (never break these)

1. **Live trading is locked.** The runtime mode is `sim | paper | live`. Code may only
   enter `live` mode when BOTH are true: the config file contains a `live_approval`
   block (date + free-text approval written by the user, never by Claude) AND the
   environment variable `TF_CONFIRM_LIVE=yes` is set at launch. Claude must never write
   the approval block, set that variable, or weaken this gate. Default mode is `sim`.
2. **Every order passes the risk manager.** There is exactly one path to the broker:
   `risk.Manager.ApproveOrder()`. No strategy, agent, or API endpoint may bypass it —
   including (especially) LLM/agentic strategies. New order paths must route through it.
3. **No lookahead.** The backtester is event-driven; a strategy sees bar N and its
   orders fill at bar N+1's open. Any change that lets a strategy see future data is a
   correctness bug of the highest severity.
4. **Secrets never enter the repo.** IBKR credentials, LLM API keys → environment
   variables or `.env` (gitignored). Never commit `.env`, tokens, or account numbers.
5. **Validation before promotion.** A strategy's lifecycle stage
   (`research → backtest → paper → probation → live → retired`) only moves forward when
   the gate criteria in `docs/ROADMAP.md` are met. Demotion on breach is automatic.
6. **Accounting honesty.** float64 is acceptable for indicators and research; before any
   live capital (Phase 4), cash/position accounting must move to fixed-point integers.
   Track this — do not silently ship float accounting to live.

## Architecture

Go **modular monolith** (split-ready, not split): one binary, `internal/` packages with
interface seams so pieces can become services later if ever needed.

```
cmd/tradeforge/     CLI entry: serve | backtest | walkforward | data
internal/domain/    core types: Bar, Order, Fill, Position, Portfolio, Signal
internal/data/      BarSource interface: CSV loader, synthetic generator, Stooq fetch/import, quality checks; (later: IBKR)
internal/strategy/  Strategy interface + registry + indicators + params (Tunable/Grid) + strategies/ (one file each; single-symbol OnBar or multi-symbol OnUniverseBar)
internal/backtest/  event-driven engine: bars → signals → risk → fills → equity curve
internal/risk/      Manager: sizing (vol-target), exposure caps, drawdown kill-switch
internal/metrics/   CAGR, Sharpe, Sortino, MaxDD, Calmar, win rate, profit factor
internal/eval/      walk-forward evaluation harness; later: deflated Sharpe, regime analysis
internal/broker/    Broker interface: SimBroker now; ibkr/ adapter later (CP Gateway)
internal/store/     SQLite run persistence (backtest/walk-forward results); `runs` CLI
internal/server/    net/http API + embedded web/ dashboard (SSE for live updates)
web/                vanilla JS dashboard, embedded via go:embed — no build step
docs/               OPTIONS (cost/stack decisions), ARCHITECTURE, ROADMAP (gates), STRATEGIES
```

- **Strategies express intent, risk sizes it.** A strategy emits `Signal{Symbol,
  Direction, TargetWeight}`; the risk manager converts intent into a sized, approved
  order (or rejects it). Strategies never compute share counts.
- **Two strategy families.** Every strategy declares `Horizon() = Short | Long`. The
  UI, evaluation reports, and eventually the ensemble allocator keep the two sections
  separate (per the project brief).
- **Registries, not wiring.** New strategies self-register in
  `internal/strategy/registry.go` via `strategy.Register(...)` — one file per strategy.

## Conventions

- Go 1.23, **stdlib + modernc.org/sqlite** (run store; see docs/OPTIONS.md
  §Dependencies). Any further new dependency needs a justification entry in
  `docs/OPTIONS.md` §Dependencies first. (Likely future exception: decimal.)
- `gofmt`-clean, `go vet ./...` clean, errors wrapped with `%w`, no `panic` outside
  `main`, contexts for anything long-running, **UTC everywhere** (`time.Time` in UTC).
- Table-driven tests. Indicators, metrics, sizing, and engine fill logic must have
  tests; a strategy without a test doesn't merge.
- Determinism: same inputs (data + seed) → same backtest output. The synthetic data
  generator takes an explicit seed.
- Frontend: vanilla ES2020, no framework, no build step; dark "trading desk" aesthetic
  (near-black background, single accent color, dense but calm). Escape all dynamic text
  before injecting into HTML.

## Verification (before calling anything done)

```powershell
go build ./... ; go vet ./... ; go test ./...
go run ./cmd/tradeforge backtest --strategy sma-cross --data testdata/sample_daily.csv
```

All must pass, and the backtest must print a sane metrics table. For UI changes: run
`go run ./cmd/tradeforge serve`, open http://localhost:8420, exercise the feature, and
screenshot it. Use the `/check` skill.

## Session discipline & model routing

- Delegate mechanical, well-specified edits to a Sonnet subagent; run an independent
  reviewer (different model, sees only the diff/files, not the reasoning) before calling
  work done; reserve the top model for planning, judgment calls, and final review.
- Pause for the user only for: destructive/irreversible actions, real scope changes, or
  decisions only they can make (e.g. anything touching the live-trading gate, spending
  money, opening accounts). Otherwise proceed and report.
- **Before ending a session, update `STATE.md`**: verified facts (with dates), any new
  general rule, and a one-line "Last session" summary with a "Next" pointer.

## Running it

```powershell
go run ./cmd/tradeforge serve                    # dashboard at http://localhost:8420
go run ./cmd/tradeforge serve --config config.json   # explicit runtime config (mode, IBKR gateway URL)
go run ./cmd/tradeforge backtest --strategy sma-cross --data testdata/sample_daily.csv
go run ./cmd/tradeforge backtest --strategy dual-momentum --universe SPY,QQQ,TLT --data-dir data
go run ./cmd/tradeforge walkforward --strategy rsi2 --data testdata/sample_daily.csv
go run ./cmd/tradeforge kfold --strategy rsi2 --data testdata/sample_daily.csv       # purged/embargoed CV, param stability only
go run ./cmd/tradeforge data synth --out testdata/sample_daily.csv --seed 42
go run ./cmd/tradeforge data import --in downloaded_spy.csv --symbol SPY --out data/SPY.csv
go run ./cmd/tradeforge runs list                                  # recent backtest/walk-forward runs
go run ./cmd/tradeforge ibkr status                                # gateway session auth/connected/competing
go run ./cmd/tradeforge ibkr accounts                              # list visible accounts
go run ./cmd/tradeforge ibkr positions --account <id>              # positions for one account
go run ./cmd/tradeforge paper run-once                             # one daily paper session (dry-run by default; needs mode "paper")
go run ./cmd/tradeforge paper status                               # recent paper sessions + their orders
```

`serve` reads `config.json` by default (see `config.example.json`; copy it to `config.json`
to customize — that file is gitignored since it may hold the user-authored `live_approval`
block). Missing config file = sim mode + default IBKR gateway URL, no setup required.

No Docker needed locally. `Dockerfile` exists for the eventual cloud deploy
(single static binary; see docs/OPTIONS.md §Hosting).

## IBKR

Paper first, always. The broker seam is `internal/broker.Broker`; the first real adapter
targets the **Client Portal Web API** (REST/WS via local gateway) because it is
language-agnostic — verify IBKR's current offering (OAuth for individuals may remove the
gateway requirement) before building it. Market data subscriptions cost real money;
default to free daily data (synthetic/Stooq) until intraday strategies need more.
