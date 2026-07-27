// Package config implements CLAUDE.md cardinal rule 1: live trading is
// locked behind a double gate. The runtime mode is one of sim (default),
// paper, or live. Mode "live" is reachable only when BOTH of the following
// hold at process start:
//
//  1. The config file contains a live_approval block — a date and a
//     free-text statement — written by the human owner of this platform.
//     Claude (or any automated agent) must NEVER generate, suggest, fill
//     in, or otherwise author the content of a real live_approval block.
//     Test code may construct one in memory or in a t.TempDir() fixture,
//     clearly as a test fixture, and never as an example of real content.
//  2. The environment variable TF_CONFIRM_LIVE is set to exactly "yes" at
//     launch.
//
// If either condition is missing, ResolveMode returns an error naming
// exactly which condition(s) failed; callers (cmd/tradeforge) must print
// that error and refuse to start rather than silently falling back to a
// safer mode. Weakening this gate — adding a bypass flag, an env var that
// skips the approval check, a "test mode" that relaxes it outside of _test.go
// fixtures, or similar — is forbidden. Any change to this file must keep
// cardinal rule 1 intact.
package config

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// Mode is the resolved runtime mode.
type Mode int

const (
	ModeSim Mode = iota
	ModePaper
	ModeLive
)

// String renders m as the lowercase form used in config files, CLI output,
// and the dashboard mode badge.
func (m Mode) String() string {
	switch m {
	case ModeSim:
		return "sim"
	case ModePaper:
		return "paper"
	case ModeLive:
		return "live"
	default:
		return "unknown"
	}
}

// LiveApproval is the user-authored record required (alongside
// TF_CONFIRM_LIVE=yes) to unlock live mode. Date and Statement are free text
// written by the human owner — Claude must never generate, suggest, or fill
// in example content for this struct outside of clearly-marked test
// fixtures.
type LiveApproval struct {
	Date      string `json:"date"`      // user-authored, e.g. "2026-08-01"
	Statement string `json:"statement"` // user-authored free text
}

// IBKRConfig holds the settings needed to reach the local Client Portal
// Gateway.
type IBKRConfig struct {
	GatewayURL string `json:"gateway_url"` // default "https://localhost:5000/v1/api"
}

// defaultGatewayURL is used whenever the config file omits ibkr.gateway_url.
const defaultGatewayURL = "https://localhost:5000/v1/api"

// PaperConfig holds the paper-trading session's configuration (see
// internal/paper and docs/OPTIONS.md §6a's daily-session design). It is
// defined here, rather than in internal/paper, to avoid an import cycle:
// internal/paper needs config.Config, and config must not import a package
// that (transitively, via internal/strategy or internal/store) could ever
// import config back.
//
// The zero value (Strategies nil, DataDir "", AccountID "") is "disabled" —
// an empty Strategies list means the paper session has nothing promoted to
// run, which is the correct, safe default for a fresh checkout. Promoting a
// strategy to the paper loop is a human act (docs/ROADMAP.md gate G1): Claude
// must never add entries to a real config.json's paper.strategies list on
// the user's behalf.
type PaperConfig struct {
	// Strategies lists registry names the user has promoted to paper
	// trading. Empty by default — nothing runs until the user edits this.
	Strategies []string `json:"strategies"`
	// DataDir is the directory containing <SYMBOL>.csv files the paper
	// session loads bars from (same layout as backtest's --data-dir).
	DataDir string `json:"data_dir"`
	// AccountID is the IBKR account to trade. Required to place real orders;
	// if empty, the session auto-discovers it when exactly one account is
	// visible to the gateway session and errors otherwise.
	AccountID string `json:"account_id"`
	// Symbols maps a single-symbol strategy's registry name to the one
	// symbol it should trade in the paper loop (single-symbol strategies
	// have no Universe() to infer this from, unlike MultiSymbol
	// strategies). Optional; only consulted for strategies that are not
	// strategy.MultiSymbol.
	Symbols map[string]string `json:"symbols,omitempty"`
}

// AgentConfig holds the (off-by-default) LLM advisory reviewer's settings —
// see internal/agent and docs/AGENT.md. This block is purely additive: it
// has no bearing on runtime mode resolution (ResolveMode above) and the
// advisor has no order path regardless of its values (internal/agent's
// package doc). The zero value (Enabled false) is "disabled", the safe
// default for a fresh checkout — enabling it is a human config edit,
// mirroring PaperConfig's "promotion is a human act" convention.
type AgentConfig struct {
	// Enabled turns the advisor on. Even when true, internal/agent.FromConfig
	// still applies the provider's own gate before making a real call — for
	// "anthropic" that means ANTHROPIC_API_KEY must be set; this flag alone
	// never causes a network call or a cost.
	Enabled bool `json:"enabled"`
	// Provider selects the advisor backend: "anthropic" (default, cloud, needs
	// an API key) or "ollama" (a local Ollama server — no API key, no
	// per-call cost, runs entirely on this machine). Empty means "anthropic"
	// for backward compatibility.
	Provider string `json:"provider,omitempty"`
	// Model overrides the provider's default model id. Empty means "use the
	// provider's own default" (see internal/agent.defaultAnthropicModel /
	// defaultOllamaModel).
	Model string `json:"model,omitempty"`
	// MaxTokens optionally lowers the advisor's per-call token cap below its
	// hard-coded ceiling. 0 means "use the provider default".
	MaxTokens int `json:"max_tokens,omitempty"`
	// BaseURL overrides the provider endpoint. Its main use is pointing the
	// "ollama" provider at a non-default local server; empty means the
	// provider default (Anthropic's API, or http://127.0.0.1:11434 for ollama).
	BaseURL string `json:"base_url,omitempty"`
}

// RiskConfig optionally TIGHTENS the paper session's risk limits relative to
// the built-in defaults (internal/risk.NewManager: 20% max position, 100%
// max gross exposure, 25% drawdown kill-switch). A zero/omitted field means
// "use the default". Validation is fail-closed and lives next to the limits
// themselves in internal/risk.NewManagerFromConfig: a value looser than the
// hard ceilings there (50% position, 100% gross, 50% drawdown) or otherwise
// nonsensical REFUSES the paper session with a loud error — it is never
// silently clamped. This block has no effect on backtests, which always run
// the fixed defaults so saved-run metrics stay comparable across time.
type RiskConfig struct {
	MaxPositionWeight float64 `json:"max_position_weight,omitempty"`
	MaxGrossExposure  float64 `json:"max_gross_exposure,omitempty"`
	MaxDrawdown       float64 `json:"max_drawdown,omitempty"`
}

// Config is the on-disk shape of config.json (see config.example.json).
type Config struct {
	Mode         string        `json:"mode"` // "sim" (default) | "paper" | "live"
	IBKR         IBKRConfig    `json:"ibkr"`
	Paper        PaperConfig   `json:"paper"`
	Agent        AgentConfig   `json:"agent,omitempty"`
	Risk         RiskConfig    `json:"risk,omitempty"`
	LiveApproval *LiveApproval `json:"live_approval,omitempty"`
}

// Load reads and parses the config file at path. A missing file is not an
// error — it returns the zero-config defaults (sim mode, default gateway
// URL) so a fresh checkout runs without any setup. A malformed JSON file, or
// one naming a mode other than sim/paper/live, is an error.
func Load(path string) (Config, error) {
	cfg := Config{
		Mode: ModeSim.String(),
		IBKR: IBKRConfig{GatewayURL: defaultGatewayURL},
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return cfg, nil
		}
		return Config{}, fmt.Errorf("config: reading %s: %w", path, err)
	}

	// Reset to defaults, then unmarshal on top so an omitted "mode" or
	// "ibkr.gateway_url" in the file still resolves to the same defaults as
	// a missing file.
	if err := json.Unmarshal(data, &cfg); err != nil {
		return Config{}, fmt.Errorf("config: parsing %s: %w", path, err)
	}
	if cfg.IBKR.GatewayURL == "" {
		cfg.IBKR.GatewayURL = defaultGatewayURL
	}
	if cfg.Mode == "" {
		cfg.Mode = ModeSim.String()
	}

	switch cfg.Mode {
	case "sim", "paper", "live":
		// ok
	default:
		return Config{}, fmt.Errorf("config: %s: unknown mode %q (must be \"sim\", \"paper\", or \"live\")", path, cfg.Mode)
	}

	return cfg, nil
}

// liveConfirmEnv is the environment variable that must be set to exactly
// "yes" (case-sensitive) for live mode to unlock. This is intentionally
// strict: "Yes", "YES", "true", "1" all fail closed.
const liveConfirmEnv = "TF_CONFIRM_LIVE"

// ResolveMode is THE GATE implementing CLAUDE.md cardinal rule 1. It turns
// c.Mode into a Mode, refusing to resolve "live" unless both required
// conditions hold. On failure the returned error names exactly which
// condition(s) are missing so an operator (or the user) can see precisely
// what to do — callers must treat this error as fatal and must not fall
// back to a different mode.
func (c Config) ResolveMode() (Mode, error) {
	switch c.Mode {
	case "", "sim":
		return ModeSim, nil
	case "paper":
		return ModePaper, nil
	case "live":
		approvalOK := c.LiveApproval != nil &&
			strings.TrimSpace(c.LiveApproval.Date) != "" &&
			strings.TrimSpace(c.LiveApproval.Statement) != ""
		// Case-sensitive: only exactly "yes" passes. "Yes", "YES", "true",
		// "1", etc. all fail closed by design — this is not a bug to be
		// made more lenient.
		envOK := os.Getenv(liveConfirmEnv) == "yes"

		if approvalOK && envOK {
			return ModeLive, nil
		}

		var missing []string
		if !approvalOK {
			missing = append(missing, "a user-written live_approval block (with non-empty date and statement) in the config file")
		}
		if !envOK {
			missing = append(missing, fmt.Sprintf("%s=yes in the environment", liveConfirmEnv))
		}
		// The zero Mode value is returned alongside the error deliberately:
		// callers must treat any non-nil error from ResolveMode as fatal and
		// must never act on the accompanying Mode value (refuse-to-start,
		// never silently-downgrade).
		return Mode(0), fmt.Errorf(
			"config: live mode requires a user-written live_approval block in the config AND %s=yes in the environment; refusing to start (missing: %s)",
			liveConfirmEnv, strings.Join(missing, "; "))
	default:
		return Mode(0), fmt.Errorf("config: unknown mode %q (must be \"sim\", \"paper\", or \"live\")", c.Mode)
	}
}
