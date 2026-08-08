package agent

import (
	"context"
	"strings"

	"tradeforge/internal/config"
)

// NullProvider is the advisor gate's disabled state: Review always returns
// Advisory{Enabled:false} with a human-readable Reason explaining why, and
// never makes a network call. It is the default Provider — cost-consciousness
// (CLAUDE.md) means the advisor costs nothing unless a human explicitly
// enables it in config AND supplies an API key.
type NullProvider struct {
	Reason string
}

func (n NullProvider) Name() string { return "null" }

func (n NullProvider) Review(ctx context.Context, in ReviewInput) (Advisory, error) {
	return Advisory{Enabled: false, Err: n.Reason}, nil
}

// FromConfig builds the Provider the paper session and server should use,
// per docs/OPTIONS.md §9 and §7's cost posture: OFF by default. It returns a
// real AnthropicProvider only when BOTH cfg.Agent.Enabled is true (a human
// edited config.json) AND getenv("ANTHROPIC_API_KEY") is non-empty (a human
// exported the key) — otherwise a NullProvider carrying the specific reason
// it's disabled, so callers (and GET /api/agent/status) can report exactly
// what's missing without ever needing a secret to do so.
//
// getenv is injected (rather than calling os.Getenv directly) so callers and
// tests can supply a hermetic environment lookup.
func FromConfig(cfg config.Config, getenv func(string) string) Provider {
	if !cfg.Agent.Enabled {
		return NullProvider{Reason: "agent advisor disabled: not enabled in config"}
	}
	switch providerName(cfg.Agent.Provider) {
	case "ollama":
		// The local provider needs no secret and makes no off-host call — its
		// only gate is Enabled (checked above). A missing/unreachable server
		// degrades at call time to Advisory{Enabled:false}, never a cost or a
		// blocked session.
		return &OllamaProvider{
			Model:     cfg.Agent.Model,
			BaseURL:   cfg.Agent.BaseURL,
			MaxTokens: cfg.Agent.MaxTokens,
		}
	default: // "anthropic"
		apiKey := ""
		if getenv != nil {
			apiKey = getenv("ANTHROPIC_API_KEY")
		}
		if apiKey == "" {
			return NullProvider{Reason: "agent advisor disabled: ANTHROPIC_API_KEY not set"}
		}
		return &AnthropicProvider{
			APIKey:    apiKey,
			Model:     cfg.Agent.Model,
			BaseURL:   cfg.Agent.BaseURL,
			MaxTokens: cfg.Agent.MaxTokens,
		}
	}
}

// providerName normalizes the configured provider id; empty means the
// backward-compatible default, "anthropic".
func providerName(p string) string {
	p = strings.ToLower(strings.TrimSpace(p))
	if p == "" {
		return "anthropic"
	}
	return p
}

// StatusReason mirrors FromConfig's enable/disable decision without
// constructing a Provider or touching a secret's value — used by GET
// /api/agent/status, which must never leak whether an API key looks valid,
// only whether one is present. reason is "" when the advisor would be
// enabled.
func StatusReason(cfg config.Config, getenv func(string) string) (enabled bool, reason, model string) {
	if !cfg.Agent.Enabled {
		return false, "agent advisor disabled: not enabled in config", ""
	}
	switch providerName(cfg.Agent.Provider) {
	case "ollama":
		effectiveModel := cfg.Agent.Model
		if effectiveModel == "" {
			effectiveModel = defaultOllamaModel
		}
		return true, "", effectiveModel
	default: // "anthropic"
		apiKey := ""
		if getenv != nil {
			apiKey = getenv("ANTHROPIC_API_KEY")
		}
		if apiKey == "" {
			return false, "agent advisor disabled: ANTHROPIC_API_KEY not set", ""
		}
		effectiveModel := cfg.Agent.Model
		if effectiveModel == "" {
			effectiveModel = defaultAnthropicModel
		}
		return true, "", effectiveModel
	}
}
