package agent

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"tradeforge/internal/config"
)

// fakeOllama builds an httptest server mimicking Ollama's /api/chat. It
// captures the last request body so tests can assert on what was sent, and
// replies with the given assistant content (verbatim) unless replyStatus is
// non-200.
func fakeOllama(t *testing.T, content string, replyStatus int, captured *string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			http.NotFound(w, r)
			return
		}
		body, _ := io.ReadAll(r.Body)
		if captured != nil {
			*captured = string(body)
		}
		w.Header().Set("Content-Type", "application/json")
		if replyStatus != 0 && replyStatus != http.StatusOK {
			w.WriteHeader(replyStatus)
			json.NewEncoder(w).Encode(map[string]any{"error": "boom"})
			return
		}
		json.NewEncoder(w).Encode(map[string]any{
			"message":           map[string]string{"role": "assistant", "content": content},
			"prompt_eval_count": 120,
			"eval_count":        40,
			"done":              true,
		})
	}))
}

func TestOllamaReview_Happy(t *testing.T) {
	var captured string
	fake := fakeOllama(t, `{"summary":"one big bet","warnings":["QQQ is 65% of equity"],"confidence":"high"}`, http.StatusOK, &captured)
	defer fake.Close()

	p := &OllamaProvider{BaseURL: fake.URL, Model: "test-model", httpClient: fake.Client()}
	adv, err := p.Review(context.Background(), ReviewInput{
		Mode: "paper", AccountEquity: 100000, AccountCurrency: "USD",
		PlannedOrders: []PlannedOrder{{Symbol: "QQQ", Side: "BUY", Qty: 130, EstimatedValue: 65000}},
	})
	if err != nil {
		t.Fatalf("Review() error = %v", err)
	}
	if !adv.Enabled {
		t.Fatalf("Enabled = false, want true; err=%q", adv.Err)
	}
	if adv.Confidence != "high" || len(adv.Warnings) != 1 || adv.Model != "test-model" {
		t.Errorf("Advisory = %+v, want confidence high, 1 warning, model test-model", adv)
	}
	if adv.TokensIn != 120 || adv.TokensOut != 40 {
		t.Errorf("tokens = %d/%d, want 120/40", adv.TokensIn, adv.TokensOut)
	}
	// The request must carry the pre-computed facts preamble, not just raw JSON.
	if !strings.Contains(captured, "% of equity") || !strings.Contains(captured, "OBJECTIVE FACTS") {
		t.Errorf("request body missing the pre-computed facts preamble: %s", captured)
	}
	if !strings.Contains(captured, `"format":"json"`) {
		t.Errorf("request did not request JSON mode: %s", captured)
	}
}

func TestOllamaReview_ParseFailureDegrades(t *testing.T) {
	fake := fakeOllama(t, `not json at all`, http.StatusOK, nil)
	defer fake.Close()

	p := &OllamaProvider{BaseURL: fake.URL, httpClient: fake.Client()}
	adv, err := p.Review(context.Background(), ReviewInput{Mode: "paper"})
	if err != nil {
		t.Fatalf("Review() error = %v, want nil (must degrade, not error)", err)
	}
	if adv.Enabled || adv.Err == "" {
		t.Errorf("Advisory = %+v, want Enabled=false with an Err on unparseable output", adv)
	}
}

func TestOllamaReview_ServerErrorDegrades(t *testing.T) {
	fake := fakeOllama(t, "", http.StatusInternalServerError, nil)
	defer fake.Close()

	p := &OllamaProvider{BaseURL: fake.URL, httpClient: fake.Client()}
	adv, err := p.Review(context.Background(), ReviewInput{Mode: "paper"})
	if err != nil {
		t.Fatalf("Review() error = %v, want nil", err)
	}
	if adv.Enabled {
		t.Errorf("Enabled = true, want false on a 5xx from the server")
	}
}

func TestOllamaReview_UnreachableDegrades(t *testing.T) {
	// Nothing listening: must degrade, never block or panic.
	p := &OllamaProvider{BaseURL: "http://127.0.0.1:1", httpClient: &http.Client{}}
	adv, err := p.Review(context.Background(), ReviewInput{Mode: "paper"})
	if err != nil {
		t.Fatalf("Review() error = %v, want nil", err)
	}
	if adv.Enabled || adv.Err == "" {
		t.Errorf("Advisory = %+v, want a degraded Advisory with an Err", adv)
	}
}

func TestNormalizeConfidence(t *testing.T) {
	cases := map[string]string{"low": "low", "LOW": "low", " High ": "high", "HIGH": "high", "medium": "medium", "": "medium", "banana": "medium"}
	for in, want := range cases {
		if got := normalizeConfidence(in); got != want {
			t.Errorf("normalizeConfidence(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestFactsPreamble(t *testing.T) {
	in := ReviewInput{
		Mode: "paper", AccountEquity: 100000, AccountCurrency: "CAD",
		PlannedOrders:      []PlannedOrder{{Symbol: "QQQ", Side: "BUY", Qty: 130, EstimatedValue: 65000}},
		DataStaleness:      "SPY is 14 days old",
		CommitteeTraceJSON: `{"selfDDTriggered":true,"capEvents":[{"symbol":"SPY"}]}`,
	}
	got := factsPreamble(in)
	for _, want := range []string{"65.0% of equity", "Largest single-order concentration: 65.0%", "base is CAD", "FX exposure", "14 days old", "de-risk TRIGGERED", "cap fired for SPY"} {
		if !strings.Contains(got, want) {
			t.Errorf("factsPreamble missing %q in:\n%s", want, got)
		}
	}

	// Flat session: no orders -> explicit NONE line, no concentration line.
	flat := factsPreamble(ReviewInput{Mode: "paper", AccountEquity: 100000, AccountCurrency: "USD"})
	if !strings.Contains(flat, "Planned orders: NONE") {
		t.Errorf("flat facts missing NONE line:\n%s", flat)
	}
	if strings.Contains(flat, "concentration:") {
		t.Errorf("flat facts should not mention concentration:\n%s", flat)
	}
}

func TestFromConfig_ProviderSelection(t *testing.T) {
	noKey := func(string) string { return "" }
	withKey := func(k string) string {
		if k == "ANTHROPIC_API_KEY" {
			return "sk-test"
		}
		return ""
	}

	// ollama: enabled, no key needed -> *OllamaProvider.
	p := FromConfig(config.Config{Agent: config.AgentConfig{Enabled: true, Provider: "ollama"}}, noKey)
	if _, ok := p.(*OllamaProvider); !ok {
		t.Errorf("ollama provider: got %T (%s), want *OllamaProvider", p, p.Name())
	}

	// ollama honors model + base_url overrides.
	p = FromConfig(config.Config{Agent: config.AgentConfig{Enabled: true, Provider: "ollama", Model: "m", BaseURL: "http://x:1"}}, noKey)
	op, ok := p.(*OllamaProvider)
	if !ok || op.Model != "m" || op.BaseURL != "http://x:1" {
		t.Errorf("ollama overrides not applied: %+v", p)
	}

	// anthropic (explicit) without key -> NullProvider.
	p = FromConfig(config.Config{Agent: config.AgentConfig{Enabled: true, Provider: "anthropic"}}, noKey)
	if p.Name() != "null" {
		t.Errorf("anthropic without key: got %s, want null", p.Name())
	}

	// default provider (empty) with key -> anthropic.
	p = FromConfig(config.Config{Agent: config.AgentConfig{Enabled: true}}, withKey)
	if _, ok := p.(*AnthropicProvider); !ok {
		t.Errorf("default provider with key: got %T, want *AnthropicProvider", p)
	}

	// disabled -> null regardless of provider.
	p = FromConfig(config.Config{Agent: config.AgentConfig{Enabled: false, Provider: "ollama"}}, noKey)
	if p.Name() != "null" {
		t.Errorf("disabled: got %s, want null", p.Name())
	}
}

func TestStatusReason_Ollama(t *testing.T) {
	enabled, reason, model := StatusReason(config.Config{Agent: config.AgentConfig{Enabled: true, Provider: "ollama"}}, func(string) string { return "" })
	if !enabled || reason != "" || model != defaultOllamaModel {
		t.Errorf("StatusReason(ollama) = (%v, %q, %q), want (true, \"\", %q)", enabled, reason, model, defaultOllamaModel)
	}
}
