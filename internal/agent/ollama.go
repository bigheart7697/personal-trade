package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// defaultOllamaModel is used when OllamaProvider.Model is empty. Chosen by the
// pre-registered evaluation in docs/AGENT_EVAL.md (see the verdict there); it
// runs fully on a 12 GB-VRAM GPU at Q4 and produces reliable JSON for this
// structured-critique task.
const defaultOllamaModel = "qwen2.5:7b-instruct"

// defaultOllamaBaseURL is Ollama's default local server address. The advisor
// only ever talks to a LOCAL model server — nothing here reaches the network
// off-host, and there is no API key (the whole point of the local provider is
// zero per-call cost and no secret to leak).
const defaultOllamaBaseURL = "http://127.0.0.1:11434"

// ollamaMaxTokensCap is the hard ceiling on generated tokens (num_predict).
// This call is a short structured critique, not an open-ended generation.
const ollamaMaxTokensCap = 1024

// defaultOllamaTimeout is generous relative to the cloud provider's 30s: a
// local model may cold-load into VRAM on the first call of a process (several
// seconds to tens of seconds), and the advisor runs at most once per paper
// session, so a slightly longer bound is a fine trade for not spuriously
// failing a cold start. Warm-call latency is far lower (measured in
// docs/AGENT_EVAL.md).
const defaultOllamaTimeout = 90 * time.Second

// ollamaSystemPrompt is tuned for a small local model: it states the
// no-order-authority guardrail, enumerates the objective checks with explicit
// numeric thresholds (smaller models follow concrete rules far better than
// vague guidance), forbids inventing facts, and demands a strict JSON object.
// Ollama is additionally asked for JSON via the request's format field, so
// this is belt-and-suspenders on the schema.
const ollamaSystemPrompt = `You are the risk-review advisor for a personal algorithmic trading platform. You have NO authority to place, modify, or cancel orders — a separate risk manager and a human operator make every such decision. Your only job is to review the plan and flag what a careful human should know before confirming.

The user message begins with an OBJECTIVE FACTS section that has ALREADY computed every number for you (each order's percent of equity, the largest concentration, any currency/FX exposure, data staleness, and committee de-risk or cap events). Trust those facts; do NOT recompute percentages yourself.

Raise one short warning for EACH fact that indicates risk:
- a concentration percent above 30% (name the symbol; above 60% is a strong warning),
- FX exposure (base currency not USD),
- stale data,
- a committee self-drawdown de-risk or a concentration-cap event.
Do NOT warn about anything the facts show is within limits — a diversified, fresh, USD plan with no concentration over 30% gets an empty warnings array and a calm summary. A concentration at or below 30% (for example 12% or 18%) is NORMAL and must NOT be warned about. Never invent a symbol or a number that is not in the facts, and never call a percentage "above 30%" unless it truly is.

Respond with ONLY a JSON object of exactly this shape and nothing else:
{"summary": "one or two sentence summary", "warnings": ["short flag", "..."], "confidence": "low" | "medium" | "high"}
Use an empty warnings array when there is nothing to flag.`

// OllamaProvider implements Provider against a local Ollama server's
// /api/chat endpoint using only the standard library. No API key, no
// off-host network call, no per-call cost. It mirrors AnthropicProvider's
// contract exactly: it never panics and degrades to
// Advisory{Enabled:false, Err: ...} (nil error) on any failure, so an
// unreachable or slow local server never blocks a paper session.
type OllamaProvider struct {
	Model     string        // defaults to defaultOllamaModel when empty
	BaseURL   string        // defaults to defaultOllamaBaseURL when empty
	Timeout   time.Duration // defaults to defaultOllamaTimeout when zero
	MaxTokens int           // lowers num_predict below the cap; 0 uses the cap

	httpClient *http.Client // overridable in tests
}

func (p *OllamaProvider) Name() string { return "ollama" }

func (p *OllamaProvider) model() string {
	if p.Model != "" {
		return p.Model
	}
	return defaultOllamaModel
}

func (p *OllamaProvider) baseURL() string {
	if p.BaseURL != "" {
		return strings.TrimRight(p.BaseURL, "/")
	}
	return defaultOllamaBaseURL
}

func (p *OllamaProvider) timeout() time.Duration {
	if p.Timeout > 0 {
		return p.Timeout
	}
	return defaultOllamaTimeout
}

func (p *OllamaProvider) client() *http.Client {
	if p.httpClient != nil {
		return p.httpClient
	}
	return &http.Client{Timeout: p.timeout()}
}

func (p *OllamaProvider) effectiveMaxTokens() int {
	if p.MaxTokens > 0 && p.MaxTokens < ollamaMaxTokensCap {
		return p.MaxTokens
	}
	return ollamaMaxTokensCap
}

// ollamaChatRequest is the body for POST /api/chat (non-streaming, JSON mode).
type ollamaChatRequest struct {
	Model    string          `json:"model"`
	Messages []ollamaMessage `json:"messages"`
	Stream   bool            `json:"stream"`
	Format   string          `json:"format,omitempty"`
	Options  ollamaOptions   `json:"options"`
}

type ollamaMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ollamaOptions struct {
	Temperature float64 `json:"temperature"`
	NumPredict  int     `json:"num_predict"`
}

// ollamaChatResponse is the subset of /api/chat's response this provider needs.
// prompt_eval_count / eval_count are Ollama's input / output token counts.
type ollamaChatResponse struct {
	Message         ollamaMessage `json:"message"`
	PromptEvalCount int           `json:"prompt_eval_count"`
	EvalCount       int           `json:"eval_count"`
	Done            bool          `json:"done"`
	Error           string        `json:"error"`
}

// Review implements Provider against the local Ollama server.
func (p *OllamaProvider) Review(ctx context.Context, in ReviewInput) (Advisory, error) {
	payload, err := json.Marshal(in)
	if err != nil {
		return Advisory{Enabled: false, Err: fmt.Sprintf("agent: marshal review input: %v", err)}, nil
	}

	reqBody := ollamaChatRequest{
		Model:  p.model(),
		Stream: false,
		Format: "json", // force JSON mode — belt-and-suspenders with the prompt
		Messages: []ollamaMessage{
			{Role: "system", Content: ollamaSystemPrompt},
			{Role: "user", Content: factsPreamble(in) + "\nRAW INPUT JSON (for reference):\n" + string(payload)},
		},
		Options: ollamaOptions{Temperature: 0, NumPredict: p.effectiveMaxTokens()},
	}
	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return Advisory{Enabled: false, Err: fmt.Sprintf("agent: marshal request: %v", err)}, nil
	}

	callCtx, cancel := context.WithTimeout(ctx, p.timeout())
	defer cancel()

	respBytes, status, err := p.do(callCtx, bodyBytes)
	if err != nil {
		return Advisory{Enabled: false, Model: p.model(), Err: fmt.Sprintf("agent: ollama request failed (is `ollama serve` running at %s?): %v", p.baseURL(), err)}, nil
	}

	var resp ollamaChatResponse
	if jsonErr := json.Unmarshal(respBytes, &resp); jsonErr != nil {
		return Advisory{Enabled: false, Model: p.model(), Err: fmt.Sprintf("agent: parse ollama response (status %d): %v", status, jsonErr)}, nil
	}
	if status != http.StatusOK || resp.Error != "" {
		msg := resp.Error
		if msg == "" {
			msg = fmt.Sprintf("status %d", status)
		}
		return Advisory{Enabled: false, Model: p.model(), Err: fmt.Sprintf("agent: ollama API error: %s", msg)}, nil
	}

	review, err := extractReviewJSON(resp.Message.Content)
	if err != nil {
		return Advisory{
			Enabled:   false,
			Model:     p.model(),
			TokensIn:  resp.PromptEvalCount,
			TokensOut: resp.EvalCount,
			Err:       fmt.Sprintf("agent: could not parse model output as JSON: %v", err),
		}, nil
	}

	return Advisory{
		Enabled:    true,
		Model:      p.model(),
		Summary:    review.Summary,
		Warnings:   review.Warnings,
		Confidence: normalizeConfidence(review.Confidence),
		TokensIn:   resp.PromptEvalCount,
		TokensOut:  resp.EvalCount,
	}, nil
}

// do performs the single HTTP call (no retry: a local server that fails is
// unlikely to succeed on an immediate retry, and the advisor is best-effort).
func (p *OllamaProvider) do(ctx context.Context, body []byte) ([]byte, int, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL()+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return nil, 0, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("content-type", "application/json")

	resp, err := p.client().Do(req)
	if err != nil {
		return nil, 0, fmt.Errorf("do request: %w", err)
	}
	defer resp.Body.Close()

	const maxBody = 1 << 20 // 1 MiB — this response is a short JSON critique
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxBody))
	if err != nil {
		return nil, resp.StatusCode, fmt.Errorf("read response body: %w", err)
	}
	return data, resp.StatusCode, nil
}

// normalizeConfidence maps a model's confidence string to the canonical
// {low,medium,high}, defaulting unknown/empty values to "medium" so a
// slightly-off token never trips the display's confidence badge.
func normalizeConfidence(c string) string {
	switch strings.ToLower(strings.TrimSpace(c)) {
	case "low":
		return "low"
	case "high":
		return "high"
	default:
		return "medium"
	}
}
