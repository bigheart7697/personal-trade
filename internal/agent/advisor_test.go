package agent

import (
	"context"
	"strings"
	"testing"

	"tradeforge/internal/config"
)

func noEnv(string) string { return "" }

func envWith(key, value string) func(string) string {
	return func(k string) string {
		if k == key {
			return value
		}
		return ""
	}
}

func TestFromConfig_DisabledByDefault(t *testing.T) {
	p := FromConfig(config.Config{}, noEnv)
	if p.Name() != "null" {
		t.Fatalf("Name() = %q, want %q", p.Name(), "null")
	}
	adv, err := p.Review(context.Background(), ReviewInput{})
	if err != nil {
		t.Fatalf("Review() error = %v, want nil", err)
	}
	if adv.Enabled {
		t.Error("Advisory.Enabled = true, want false when agent.enabled is false")
	}
	if !strings.Contains(adv.Err, "not enabled in config") {
		t.Errorf("Advisory.Err = %q, want it to explain the config gate", adv.Err)
	}
}

func TestFromConfig_EnabledButNoAPIKey(t *testing.T) {
	cfg := config.Config{Agent: config.AgentConfig{Enabled: true}}
	p := FromConfig(cfg, noEnv)
	if p.Name() != "null" {
		t.Fatalf("Name() = %q, want %q (no API key)", p.Name(), "null")
	}
	adv, _ := p.Review(context.Background(), ReviewInput{})
	if adv.Enabled {
		t.Error("Advisory.Enabled = true, want false with no ANTHROPIC_API_KEY")
	}
	if !strings.Contains(adv.Err, "ANTHROPIC_API_KEY") {
		t.Errorf("Advisory.Err = %q, want it to mention the missing env var", adv.Err)
	}
}

func TestFromConfig_EnabledWithAPIKeyReturnsAnthropicProvider(t *testing.T) {
	cfg := config.Config{Agent: config.AgentConfig{Enabled: true, Model: "claude-haiku-4-5", MaxTokens: 512}}
	p := FromConfig(cfg, envWith("ANTHROPIC_API_KEY", testAPIKey))
	if p.Name() != "anthropic" {
		t.Fatalf("Name() = %q, want %q", p.Name(), "anthropic")
	}
	ap, ok := p.(*AnthropicProvider)
	if !ok {
		t.Fatalf("FromConfig returned %T, want *AnthropicProvider", p)
	}
	if ap.APIKey != testAPIKey {
		t.Errorf("APIKey = %q, want the env var value", ap.APIKey)
	}
	if ap.Model != "claude-haiku-4-5" {
		t.Errorf("Model = %q, want the config value", ap.Model)
	}
	if ap.MaxTokens != 512 {
		t.Errorf("MaxTokens = %d, want 512", ap.MaxTokens)
	}
}

func TestStatusReason_MirrorsFromConfig(t *testing.T) {
	tests := []struct {
		name        string
		cfg         config.Config
		getenv      func(string) string
		wantEnabled bool
		wantReason  string
	}{
		{
			name:        "disabled in config",
			cfg:         config.Config{},
			getenv:      noEnv,
			wantEnabled: false,
			wantReason:  "not enabled in config",
		},
		{
			name:        "enabled but no key",
			cfg:         config.Config{Agent: config.AgentConfig{Enabled: true}},
			getenv:      noEnv,
			wantEnabled: false,
			wantReason:  "ANTHROPIC_API_KEY",
		},
		{
			name:        "enabled with key",
			cfg:         config.Config{Agent: config.AgentConfig{Enabled: true}},
			getenv:      envWith("ANTHROPIC_API_KEY", testAPIKey),
			wantEnabled: true,
			wantReason:  "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			enabled, reason, model := StatusReason(tt.cfg, tt.getenv)
			if enabled != tt.wantEnabled {
				t.Errorf("enabled = %v, want %v", enabled, tt.wantEnabled)
			}
			if tt.wantReason != "" && !strings.Contains(reason, tt.wantReason) {
				t.Errorf("reason = %q, want it to contain %q", reason, tt.wantReason)
			}
			if tt.wantEnabled && model == "" {
				t.Error("model = \"\", want a default model name when enabled")
			}
			if reason != "" && strings.Contains(reason, testAPIKey) {
				t.Errorf("reason leaked the API key: %q", reason)
			}
		})
	}
}
