# TradeForge

A personal algorithmic trading platform for a single user. Go backend, zero-build
vanilla-JS dashboard, IBKR paper-then-live (double-gated).

## What this is

TradeForge is a **strategy laboratory first, broker second**. Strategies are written,
backtested on historical data, and evaluated under deliberately strict validation —
walk-forward out-of-sample testing, a deflated Sharpe ratio that accounts for how many
variants were tried, regime breakdowns, and buy-and-hold/random-entry baselines — before
they ever see a brokerage account. The honest-evaluation philosophy: a strategy's
backtest number is a claim, not a fact, and every tool in this repo exists to make that
claim harder to fake. Promotion from research to paper to live follows explicit,
recorded gates (`docs/ROADMAP.md`); nothing is promoted on backtest numbers alone, and
live trading requires the user's own explicit, dated approval — never an automated or
agentic decision.

## Quickstart

Requires Go 1.23+.

```powershell
git clone <this repo>
cd TradeForge
go build ./...

# Generate synthetic daily-bar data (deterministic given a seed — no network needed)
go run ./cmd/tradeforge data synth --out testdata/sample_daily.csv --seed 42

# Run your first backtest
go run ./cmd/tradeforge backtest --strategy sma-cross --data testdata/sample_daily.csv

# Open the dashboard
go run ./cmd/tradeforge serve
# → http://localhost:8420
```

The dashboard lets you pick a strategy, run a backtest against synthetic data, and see
metrics + an equity curve render live in the browser — no data files required to start.

## Real data

Free daily bars come from Stooq. Stooq's automated CSV endpoint is fronted by a
JavaScript bot-check that plain HTTP requests can't pass, so the reliable path is a
**manual browser download**:

1. In a browser, visit Stooq's historical-data page for your symbol and download the CSV.
2. Import it into TradeForge's canonical format:
   ```powershell
   go run ./cmd/tradeforge data import --in downloaded_spy.csv --symbol SPY --out data/SPY.csv
   ```
3. `data import` (and `data fetch`, if the Stooq endpoint happens to be reachable
   unattended) both run the **quality checker** automatically — it flags calendar gaps
   ≥4 weekdays, single-day moves >25%, and zero-volume rows, so a data problem surfaces
   before it corrupts a backtest. Run it standalone any time with `data check`.

`data/` is gitignored — vendor data terms mean it should never be committed.

## Evaluating strategies

Three evaluation modes, increasingly strict:

- **`backtest`** — a single run over the full data window. Fast, useful for sanity
  checks and iteration, but a single train/test split proves nothing about robustness.
- **`walkforward`** — rolling train-window → test-window folds (parameters are chosen
  only on train data, then scored on the following out-of-sample window, stitched end
  to end). This is the number that matters for promotion.
- **`kfold`** — purged & embargoed cross-validation across the whole series, used as a
  parameter-stability check alongside walk-forward (it can leak a little information
  within a fold by design, so its Sharpe is never itself a promotion number).

Every evaluation report also carries:

- **DSR (deflated Sharpe ratio)** — the probability the observed Sharpe is real once you
  account for how many parameter combinations were tried; it is brutal on purpose.
- **Benchmark** — the strategy's Sharpe/CAGR/drawdown against SPY buy-and-hold over the
  same out-of-sample window.
- **Regimes** — the same out-of-sample period sliced into bull/bear/chop, so a strategy
  that only works in one regime can't hide behind an aggregate number.
- **Random-entry baseline** — the same exposure (same number and length of holding
  periods) placed at random, many times, so a strategy has to beat plain luck, not just
  "being in the market."

**No strategy is promoted on backtest numbers alone** — see `docs/ROADMAP.md`'s
promotion gates (G1–G4), which require walk-forward OOS performance, a passing DSR, and,
further down the lifecycle, months of live paper-trading evidence and the user's own
explicit sign-off.

## The strategy catalog

Full catalog with status and design notes: `docs/STRATEGIES.md`. Currently implemented:

| Name | Family | One-liner |
|---|---|---|
| `rsi2` | Short-term | Connors RSI(2) mean reversion: buy oversold dips in uptrends, exit on RSI2 > 60 or a time stop |
| `boll-snapback` | Short-term | Fade closes below the lower Bollinger band in quiet-vol regimes |
| `pairs-rv` | Short-term | Long-only relative-value rotation between two symbols on a mean-reverting log-price spread z-score |
| `turn-of-month` | Short-term | Turn-of-month seasonality: long across the month boundary, flat mid-month |
| `ibs` | Short-term | Internal Bar Strength mean reversion: buy weak closes (in uptrends) on a bar's own intraday range, exit strong closes |
| `ml-logit` | Short-term / ML | Logistic regression on six technical features predicting next-bar direction, refit monthly on a rolling window |
| `ml-knn` | Short-term / ML | k-nearest-neighbor pattern matcher over recent return shapes; long when similar histories resolved up |
| `sma-cross` | Long-term | 50/200 SMA trend following with vol-targeted sizing |
| `donchian` | Long-term | 55-day channel breakout, 20-day exit |
| `dual-momentum` | Long-term | Antonacci-style relative + absolute momentum across an ETF universe, rotating to a defensive asset or cash |
| `vol-target` | Long-term | Constant-volatility equity exposure, rebalanced monthly |
| `tsmom` | Long-term | 12-1 time-series momentum, vol-scaled sizing, monthly rebalance |
| `ml-boost` | Long-term / ML | Hand-rolled gradient-boosted decision stumps regressing next-bar return, refit quarterly |
| `ensemble-lite` | Meta | Volatility-weighted meta-allocator over a fixed roster; promotion gated on ≥2 members independently passing gate G2 |
| `agent-committee` | Meta / agentic | Allocates across five member strategies by recent virtual track record, with regime gating, concentration caps, self-de-risking, and a persisted deliberation trace (see `docs/AGENT.md`) |

List what's registered at any time with `tradeforge backtest --list`.

## Paper trading

Paper trading routes real signals through the real risk manager against IBKR's **paper**
brokerage account — the last stop before any of this touches real money. Setup
(distilled from `docs/OPTIONS.md` §6a — see it for the full research trail):

1. **Download and run the IBKR Client Portal Gateway** (a local Java process; IBKR
   requires it for individual/retail API access — there is no gateway-free OAuth path
   for individuals as of this writing). Start it; it listens on `https://localhost:5000`.
2. **Log in via browser**: open `https://localhost:5000`, and sign in using the **Paper**
   toggle/credentials (not your live credentials). 2FA is required on every login, paper
   included — this is deliberate on IBKR's side and is not automated here.
3. Confirm the session from the CLI:
   ```powershell
   go run ./cmd/tradeforge ibkr status
   ```
   Expect `Authenticated: true`. A freshly created paper account can take up to ~24h
   (the next overnight reset) before the API session actually establishes even though
   browser login succeeds — this is normal IBKR provisioning lag, not a bug here.
4. **Promote a strategy** by adding its registry name to config.json's `paper.strategies`
   list (copy `config.example.json` to `config.json` first). This is a human act by
   design — TradeForge never adds itself to the paper loop.
5. **Dry-run first, always**:
   ```powershell
   go run ./cmd/tradeforge paper run-once
   ```
   This computes signals, runs them through the risk manager, and prints exactly what
   *would* happen — it places nothing. Only once you've read and understood the output:
   ```powershell
   go run ./cmd/tradeforge paper run-once --execute
   ```
6. **Daily-session model**: because TradeForge trades daily bars (signal at close, fill
   at next open), paper trading does not need a 24/7 connection — one short scheduled
   run per trading day (`paper run-once`) is the whole design. This sidesteps the
   Client Portal Gateway's known session fragility (idle timeout in minutes, hard
   session limit around 24h) entirely.
7. Review recorded sessions and orders any time with `tradeforge paper status`, or in
   the dashboard's Paper Trading panel.

## Safety model

TradeForge's rules exist because this platform eventually touches real money, and the
cost of a mistake there is not the same as the cost of a mistake in a backtest.

- **Live trading is locked behind a double gate.** The runtime mode is `sim` (default),
  `paper`, or `live`. Mode `live` only unlocks when the config file contains a
  user-written approval block (a date and a free-text statement — never generated by an
  agent) **and** an environment variable is explicitly set at launch. Neither condition
  can be satisfied by code; both are acts only the user can perform.
- **Every order passes through one risk manager**, no exceptions. There is exactly one
  path from a strategy's signal to a broker order, and it always goes through sizing,
  exposure caps, and a drawdown kill-switch — including for any future LLM/agentic
  strategy.
- **Dry-run is the default** everywhere an order could be placed. Paper trading prints
  its intended orders and executes nothing unless you explicitly pass `--execute`.
- **Nothing is promoted without evidence and a human decision.** A strategy's lifecycle
  (research → backtest → paper → probation → live → retired) only advances when the
  gate criteria in `docs/ROADMAP.md` are met, and the paper→probation gate additionally
  requires the user's own recorded approval — not an automatic pass on numbers alone.
  Any breach along the way demotes the strategy automatically.

## Project layout

```
cmd/tradeforge/     CLI entry point: serve | backtest | walkforward | kfold | data | runs | ibkr | paper
internal/domain/    core types — Bar, Order, Fill, Position, Portfolio, Signal, fixed-point Money
internal/data/      BarSource implementations: CSV loader, synthetic generator, Stooq fetch/import, quality checks
internal/strategy/  Strategy interface, registry, indicators, tunable params, and the strategies themselves
internal/backtest/  event-driven engine — bars → signals → risk → fills → equity curve, single- and multi-symbol
internal/risk/      the risk manager — vol-target sizing, exposure caps, drawdown kill-switch
internal/metrics/   CAGR, Sharpe, Sortino, MaxDD, Calmar, win rate, profit factor, deflated Sharpe
internal/eval/      walk-forward and purged K-fold evaluation harnesses, regime breakdown, random-entry baseline
internal/broker/    Broker interface — SimBroker today; internal/broker/ibkr is the read-only Client Portal client
internal/paper/     the daily paper-trading session: strategy signals → risk manager → IBKR paper orders
internal/store/     SQLite persistence for backtest/walk-forward/K-fold runs and paper-trading sessions
internal/server/    the HTTP API plus the embedded dashboard (internal/server/web)
internal/config/    runtime config and the live-trading mode gate
web/                (embedded from internal/server/web) vanilla JS dashboard, no build step
docs/               OPTIONS.md (stack/cost decisions), ARCHITECTURE.md (system design),
                    ROADMAP.md (phases and promotion gates), STRATEGIES.md (the full catalog)
```

See also `CLAUDE.md` for the rules this codebase is developed under, and `STATE.md` for
a running log of verified facts and session history.
