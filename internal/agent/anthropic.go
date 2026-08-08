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

// defaultAnthropicModel is used when AnthropicProvider.Model is empty.
// claude-haiku-4-5 is the cheapest-capable current tier (per docs/OPTIONS.md
// §7 and the claude-api skill's pricing table) — right-sized for a single,
// low-frequency, bounded (MaxTokens-capped) risk-review call per paper
// session, where cost-consciousness is a design goal (CLAUDE.md).
const defaultAnthropicModel = "claude-haiku-4-5"

// anthropicMaxTokensCap is a hard ceiling on MaxTokens regardless of what a
// config file requests — this call is a short structured critique, not an
// open-ended generation, and the cap keeps a misconfigured value from
// blowing the "cents/day" cost budget.
const anthropicMaxTokensCap = 1024

// anthropicVersion is the Messages API version header value.
const anthropicVersion = "2023-06-01"

// defaultAnthropicBaseURL is the production Messages API endpoint.
const defaultAnthropicBaseURL = "https://api.anthropic.com"

// defaultAnthropicTimeout bounds a single attempt (including its one retry
// per the retry policy below).
const defaultAnthropicTimeout = 30 * time.Second

// anthropicRetryBackoff is the fixed delay before the one retry on
// 429/5xx/timeout. A var (not const) so tests can shrink it.
var anthropicRetryBackoff = 2 * time.Second

// AnthropicProvider implements Provider against Anthropic's Messages API
// using only the standard library (net/http) — no SDK dependency, per
// docs/OPTIONS.md §9's hand-rolled-loop decision.
//
// APIKey must come from the environment (ANTHROPIC_API_KEY) — never from a
// config file, never logged, never included in an Advisory or error string
// (see the tests in anthropic_test.go asserting exactly this).
type AnthropicProvider struct {
	APIKey  string
	Model   string        // defaults to defaultAnthropicModel when empty
	BaseURL string        // defaults to defaultAnthropicBaseURL when empty; overridable for tests
	Timeout time.Duration // defaults to defaultAnthropicTimeout when zero

	// MaxTokens optionally lowers the request's max_tokens below the hard
	// cap (anthropicMaxTokensCap); zero (the default) uses the cap. This can
	// only ever LOWER the effective limit, never raise it past the cap —
	// see effectiveMaxTokens.
	MaxTokens int

	// httpClient is overridable in tests; defaults to a client built from
	// Timeout.
	httpClient *http.Client
}

func (p *AnthropicProvider) Name() string { return "anthropic" }

func (p *AnthropicProvider) model() string {
	if p.Model != "" {
		return p.Model
	}
	return defaultAnthropicModel
}

func (p *AnthropicProvider) baseURL() string {
	if p.BaseURL != "" {
		return strings.TrimRight(p.BaseURL, "/")
	}
	return defaultAnthropicBaseURL
}

func (p *AnthropicProvider) timeout() time.Duration {
	if p.Timeout > 0 {
		return p.Timeout
	}
	return defaultAnthropicTimeout
}

func (p *AnthropicProvider) client() *http.Client {
	if p.httpClient != nil {
		return p.httpClient
	}
	return &http.Client{Timeout: p.timeout()}
}

// effectiveMaxTokens returns p.MaxTokens when it is set and smaller than the
// hard cap, otherwise the cap. A configured value can only lower the
// request's max_tokens, never raise it — cost-consciousness (CLAUDE.md) is
// enforced here regardless of what a config file requests.
func (p *AnthropicProvider) effectiveMaxTokens() int {
	if p.MaxTokens > 0 && p.MaxTokens < anthropicMaxTokensCap {
		return p.MaxTokens
	}
	return anthropicMaxTokensCap
}

// anthropicSystemPrompt is the fixed system prompt for the advisory review.
// It states the "no order authority" guardrail explicitly so the model's own
// framing of its output matches the platform's contract.
const anthropicSystemPrompt = `You are the risk-review advisor for a personal algorithmic trading platform. You have NO authority to place, modify, or cancel orders — a separate risk manager and a human operator make all such decisions; your job is only to review the planned orders below and flag anything a careful human reviewer would want to know before confirming. Consider: concentration risk, unusual position sizes relative to account equity, data staleness, and anything inconsistent with the recent session history provided.

Respond ONLY with a JSON object of exactly this shape, no other text:
{"summary": "one or two sentence summary", "warnings": ["short flag 1", "short flag 2"], "confidence": "low"|"medium"|"high"}`

// anthropicMessageRequest is the request body for POST /v1/messages.
type anthropicMessageRequest struct {
	Model     string                  `json:"model"`
	MaxTokens int                     `json:"max_tokens"`
	System    string                  `json:"system,omitempty"`
	Messages  []anthropicMessageParam `json:"messages"`
}

type anthropicMessageParam struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// anthropicMessageResponse is the subset of the Messages API response shape
// this provider needs.
type anthropicMessageResponse struct {
	Content []anthropicContentBlock `json:"content"`
	Usage   anthropicUsage          `json:"usage"`
}

type anthropicContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type anthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// anthropicErrorEnvelope is the API's error response shape.
type anthropicErrorEnvelope struct {
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// modelReviewJSON is the JSON shape the system prompt asks the model to
// return; parsed out of the response text (tolerating surrounding prose).
type modelReviewJSON struct {
	Summary    string   `json:"summary"`
	Warnings   []string `json:"warnings"`
	Confidence string   `json:"confidence"`
}

// Review implements Provider. It never panics and never returns advisory
// content mixed with the API key: on any failure it returns
// Advisory{Enabled:false, Err: ...} with a nil error — the error return is
// reserved for programmer-error cases that should never occur in practice
// (there are none in this implementation; it always returns nil error).
func (p *AnthropicProvider) Review(ctx context.Context, in ReviewInput) (Advisory, error) {
	payload, err := json.Marshal(in)
	if err != nil {
		return Advisory{Enabled: false, Err: fmt.Sprintf("agent: marshal review input: %v", err)}, nil
	}

	reqBody := anthropicMessageRequest{
		Model:     p.model(),
		MaxTokens: p.effectiveMaxTokens(),
		System:    anthropicSystemPrompt,
		Messages: []anthropicMessageParam{
			{Role: "user", Content: string(payload)},
		},
	}
	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return Advisory{Enabled: false, Err: fmt.Sprintf("agent: marshal request: %v", err)}, nil
	}

	callCtx, cancel := context.WithTimeout(ctx, p.timeout())
	defer cancel()

	respBytes, status, err := p.doWithRetry(callCtx, bodyBytes)
	if err != nil {
		return Advisory{Enabled: false, Model: p.model(), Err: sanitizeErr(err, p.APIKey)}, nil
	}
	if status != http.StatusOK {
		msg := parseErrorMessage(respBytes)
		return Advisory{Enabled: false, Model: p.model(), Err: fmt.Sprintf("agent: anthropic API returned %d: %s", status, sanitizeText(msg, p.APIKey))}, nil
	}

	var resp anthropicMessageResponse
	if err := json.Unmarshal(respBytes, &resp); err != nil {
		return Advisory{Enabled: false, Model: p.model(), Err: fmt.Sprintf("agent: parse response: %v", err)}, nil
	}

	var text strings.Builder
	for _, block := range resp.Content {
		if block.Type == "text" {
			text.WriteString(block.Text)
		}
	}

	review, err := extractReviewJSON(text.String())
	if err != nil {
		return Advisory{
			Enabled:   false,
			Model:     p.model(),
			TokensIn:  resp.Usage.InputTokens,
			TokensOut: resp.Usage.OutputTokens,
			Err:       fmt.Sprintf("agent: could not parse model output as JSON: %v", err),
		}, nil
	}

	return Advisory{
		Enabled:    true,
		Model:      p.model(),
		Summary:    review.Summary,
		Warnings:   review.Warnings,
		Confidence: review.Confidence,
		TokensIn:   resp.Usage.InputTokens,
		TokensOut:  resp.Usage.OutputTokens,
	}, nil
}

// doWithRetry performs the HTTP call, retrying exactly once on 429, 5xx, or a
// transport-level error (including timeout), with a fixed 2s backoff.
func (p *AnthropicProvider) doWithRetry(ctx context.Context, body []byte) (respBody []byte, status int, err error) {
	respBody, status, err = p.do(ctx, body)
	if err == nil && status != http.StatusTooManyRequests && status < http.StatusInternalServerError {
		return respBody, status, nil
	}

	// One retry, 2s backoff — bail early if the context won't survive the wait.
	select {
	case <-ctx.Done():
		if err != nil {
			return respBody, status, err
		}
		return respBody, status, ctx.Err()
	case <-time.After(anthropicRetryBackoff):
	}

	return p.do(ctx, body)
}

func (p *AnthropicProvider) do(ctx context.Context, body []byte) ([]byte, int, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL()+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, 0, fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("content-type", "application/json")
	req.Header.Set("x-api-key", p.APIKey)
	req.Header.Set("anthropic-version", anthropicVersion)

	resp, err := p.client().Do(req)
	if err != nil {
		return nil, 0, fmt.Errorf("do request: %w", err)
	}
	defer resp.Body.Close()

	const maxBody = 1 << 20 // 1 MiB — this response is always a short JSON critique
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxBody))
	if err != nil {
		return nil, resp.StatusCode, fmt.Errorf("read response body: %w", err)
	}
	return data, resp.StatusCode, nil
}

// parseErrorMessage extracts the human-readable message from an Anthropic
// error envelope, falling back to the raw body (truncated) if it doesn't
// parse as the expected shape.
func parseErrorMessage(body []byte) string {
	var env anthropicErrorEnvelope
	if err := json.Unmarshal(body, &env); err == nil && env.Error.Message != "" {
		return env.Error.Message
	}
	s := string(body)
	const maxLen = 500
	if len(s) > maxLen {
		s = s[:maxLen] + "..."
	}
	return s
}

// extractReviewJSON parses the model's expected {"summary","warnings",
// "confidence"} object out of raw, tolerating any surrounding prose by
// extracting the first balanced {...} block before parsing.
func extractReviewJSON(raw string) (modelReviewJSON, error) {
	start := strings.IndexByte(raw, '{')
	if start < 0 {
		return modelReviewJSON{}, fmt.Errorf("no JSON object found in model output")
	}
	depth := 0
	end := -1
	for i := start; i < len(raw); i++ {
		switch raw[i] {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				end = i
			}
		}
		if end >= 0 {
			break
		}
	}
	if end < 0 {
		return modelReviewJSON{}, fmt.Errorf("unbalanced JSON object in model output")
	}

	var review modelReviewJSON
	if err := json.Unmarshal([]byte(raw[start:end+1]), &review); err != nil {
		return modelReviewJSON{}, fmt.Errorf("unmarshal extracted JSON: %w", err)
	}
	return review, nil
}

// sanitizeErr renders err as a string with apiKey scrubbed, defense in depth
// against the key ever leaking into an Advisory or a log line (the key is
// never intentionally interpolated into an error, but a future change
// touching this file must not regress that guarantee).
func sanitizeErr(err error, apiKey string) string {
	return sanitizeText(fmt.Sprintf("agent: anthropic request failed: %v", err), apiKey)
}

func sanitizeText(s, apiKey string) string {
	if apiKey == "" {
		return s
	}
	return strings.ReplaceAll(s, apiKey, "[REDACTED]")
}
