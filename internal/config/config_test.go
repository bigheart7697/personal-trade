package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeConfig(t *testing.T, contents string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatalf("writing test config: %v", err)
	}
	return path
}

func TestLoad_MissingFileReturnsDefaults(t *testing.T) {
	path := filepath.Join(t.TempDir(), "does-not-exist.json")
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load() error = %v, want nil", err)
	}
	if cfg.Mode != "sim" {
		t.Errorf("Mode = %q, want \"sim\"", cfg.Mode)
	}
	if cfg.IBKR.GatewayURL != defaultGatewayURL {
		t.Errorf("IBKR.GatewayURL = %q, want %q", cfg.IBKR.GatewayURL, defaultGatewayURL)
	}
	if cfg.LiveApproval != nil {
		t.Errorf("LiveApproval = %+v, want nil", cfg.LiveApproval)
	}
}

func TestLoad_UnknownModeErrors(t *testing.T) {
	path := writeConfig(t, `{"mode": "yolo"}`)
	_, err := Load(path)
	if err == nil {
		t.Fatal("Load() error = nil, want error for unknown mode")
	}
	if !strings.Contains(err.Error(), "yolo") {
		t.Errorf("Load() error = %q, want it to mention the bad mode", err.Error())
	}
}

func TestLoad_MalformedJSONErrors(t *testing.T) {
	path := writeConfig(t, `{not valid json`)
	_, err := Load(path)
	if err == nil {
		t.Fatal("Load() error = nil, want error for malformed JSON")
	}
}

func TestLoad_DefaultGatewayURLWhenOmitted(t *testing.T) {
	path := writeConfig(t, `{"mode": "paper"}`)
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if cfg.IBKR.GatewayURL != defaultGatewayURL {
		t.Errorf("IBKR.GatewayURL = %q, want %q", cfg.IBKR.GatewayURL, defaultGatewayURL)
	}
}

func TestLoad_CustomGatewayURLPreserved(t *testing.T) {
	path := writeConfig(t, `{"mode": "sim", "ibkr": {"gateway_url": "https://localhost:9999/v1/api"}}`)
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if cfg.IBKR.GatewayURL != "https://localhost:9999/v1/api" {
		t.Errorf("IBKR.GatewayURL = %q, want the custom URL preserved", cfg.IBKR.GatewayURL)
	}
}

func TestResolveMode_Table(t *testing.T) {
	// testApproval is a TEST-ONLY fixture. It must never be treated as an
	// example of real user-authored content — production live_approval
	// blocks are written exclusively by the human owner, never by Claude or
	// any code path in this repo.
	testApproval := &LiveApproval{Date: "2099-01-01", Statement: "test-fixture-approval"}

	tests := []struct {
		name       string
		cfg        Config
		envValue   string // "" means unset
		envSet     bool
		wantMode   Mode
		wantErr    bool
		wantErrHas []string // substrings that must all appear in the error
	}{
		{
			name:     "empty mode defaults to sim",
			cfg:      Config{Mode: ""},
			wantMode: ModeSim,
		},
		{
			name:     "sim never requires anything",
			cfg:      Config{Mode: "sim"},
			wantMode: ModeSim,
		},
		{
			name:     "paper never requires approval",
			cfg:      Config{Mode: "paper"},
			wantMode: ModePaper,
		},
		{
			name:     "paper never requires env var even if set oddly",
			cfg:      Config{Mode: "paper"},
			envSet:   true,
			envValue: "no",
			wantMode: ModePaper,
		},
		{
			name:     "live with approval and env=yes succeeds",
			cfg:      Config{Mode: "live", LiveApproval: testApproval},
			envSet:   true,
			envValue: "yes",
			wantMode: ModeLive,
		},
		{
			name:       "live with approval but env unset fails naming env",
			cfg:        Config{Mode: "live", LiveApproval: testApproval},
			envSet:     false,
			wantErr:    true,
			wantErrHas: []string{"TF_CONFIRM_LIVE=yes"},
		},
		{
			name:       "live with approval but env empty fails naming env",
			cfg:        Config{Mode: "live", LiveApproval: testApproval},
			envSet:     true,
			envValue:   "",
			wantErr:    true,
			wantErrHas: []string{"TF_CONFIRM_LIVE=yes"},
		},
		{
			name:       "live with approval but env=no fails naming env",
			cfg:        Config{Mode: "live", LiveApproval: testApproval},
			envSet:     true,
			envValue:   "no",
			wantErr:    true,
			wantErrHas: []string{"TF_CONFIRM_LIVE=yes"},
		},
		{
			name:       "live with approval but env=YES (wrong case) fails naming env",
			cfg:        Config{Mode: "live", LiveApproval: testApproval},
			envSet:     true,
			envValue:   "YES",
			wantErr:    true,
			wantErrHas: []string{"TF_CONFIRM_LIVE=yes"},
		},
		{
			name:       "live with env but nil approval fails naming approval",
			cfg:        Config{Mode: "live", LiveApproval: nil},
			envSet:     true,
			envValue:   "yes",
			wantErr:    true,
			wantErrHas: []string{"live_approval"},
		},
		{
			name:       "live with env but empty date fails naming approval",
			cfg:        Config{Mode: "live", LiveApproval: &LiveApproval{Date: "", Statement: "test-fixture-approval"}},
			envSet:     true,
			envValue:   "yes",
			wantErr:    true,
			wantErrHas: []string{"live_approval"},
		},
		{
			name:       "live with env but empty statement fails naming approval",
			cfg:        Config{Mode: "live", LiveApproval: &LiveApproval{Date: "2099-01-01", Statement: ""}},
			envSet:     true,
			envValue:   "yes",
			wantErr:    true,
			wantErrHas: []string{"live_approval"},
		},
		{
			name:       "live with env but whitespace-only date and statement fails naming approval",
			cfg:        Config{Mode: "live", LiveApproval: &LiveApproval{Date: "   ", Statement: "   "}},
			envSet:     true,
			envValue:   "yes",
			wantErr:    true,
			wantErrHas: []string{"live_approval"},
		},
		{
			name:       "live with neither approval nor env fails naming both",
			cfg:        Config{Mode: "live"},
			envSet:     false,
			wantErr:    true,
			wantErrHas: []string{"live_approval", "TF_CONFIRM_LIVE=yes"},
		},
		{
			name:       "unknown mode errors",
			cfg:        Config{Mode: "yolo"},
			wantErr:    true,
			wantErrHas: []string{"yolo"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.envSet {
				t.Setenv(liveConfirmEnv, tt.envValue)
			} else {
				// Ensure the variable is unset even if the host environment
				// happens to have it set (t.Setenv guarantees restoration).
				if _, ok := os.LookupEnv(liveConfirmEnv); ok {
					t.Setenv(liveConfirmEnv, "")
					os.Unsetenv(liveConfirmEnv)
				}
			}

			mode, err := tt.cfg.ResolveMode()

			if tt.wantErr {
				if err == nil {
					t.Fatalf("ResolveMode() error = nil, want error")
				}
				for _, sub := range tt.wantErrHas {
					if !strings.Contains(err.Error(), sub) {
						t.Errorf("ResolveMode() error = %q, want it to contain %q", err.Error(), sub)
					}
				}
				return
			}

			if err != nil {
				t.Fatalf("ResolveMode() error = %v, want nil", err)
			}
			if mode != tt.wantMode {
				t.Errorf("ResolveMode() mode = %v, want %v", mode, tt.wantMode)
			}
		})
	}
}

func TestMode_String(t *testing.T) {
	tests := []struct {
		mode Mode
		want string
	}{
		{ModeSim, "sim"},
		{ModePaper, "paper"},
		{ModeLive, "live"},
	}
	for _, tt := range tests {
		if got := tt.mode.String(); got != tt.want {
			t.Errorf("Mode(%d).String() = %q, want %q", tt.mode, got, tt.want)
		}
	}
}
