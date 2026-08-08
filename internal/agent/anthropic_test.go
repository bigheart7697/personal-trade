package agent

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

const testAPIKey = "sk-ant-test-super-secret-do-not-leak"

func TestAnthropicProvider_Review_Success(t *testing.T) {
	var gotAPIKey, gotVersion, gotContentType string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAPIKey = r.Header.Get("x-api-key")
		gotVersion = r.Header.Get("anthropic-version")
		gotContentType = r.Header.Get("content-type")

		resp := anthropicMessageResponse{
			Content: []anthropicContentBlock{
				{Type: "text", Text: `{"summary": "looks fine", "warnings": ["concentration: 2 orders add QQQ exposure"], "confidence": "medium"}`},
			},
			Usage: anthropicUsage{InputTokens: 120, OutputTokens: 40},
		}
		w.Header().Set("content-type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	p := &AnthropicProvider{APIKey: testAPIKey, BaseURL: srv.URL}
	adv, err := p.Review(context.Background(), ReviewInput{Mode: "paper"})
	if err != nil {
		t.Fatalf("Review() error = %v, want nil", err)
	}
	if !adv.Enabled {
		t.Fatalf("Advisory.Enabled = false, want true; Err=%q", adv.Err)
	}
	if adv.Summary != "looks fine" {
		t.Errorf("Summary = %q, want %q", adv.Summary, "looks fine")
	}
	if len(adv.Warnings) != 1 || adv.Warnings[0] != "concentration: 2 orders add QQQ exposure" {
		t.Errorf("Warnings = %v, want one concentration warning", adv.Warnings)
	}
	if adv.Confidence != "medium" {
		t.Errorf("Confidence = %q, want %q", adv.Confidence, "medium")
	}
	if adv.TokensIn != 120 || adv.TokensOut != 40 {
		t.Errorf("Tokens = (%d,%d), want (120,40)", adv.TokensIn, adv.TokensOut)
	}

	if gotAPIKey != testAPIKey {
		t.Errorf("x-api-key header = %q, want %q", gotAPIKey, testAPIKey)
	}
	if gotVersion != anthropicVersion {
		t.Errorf("anthropic-version header = %q, want %q", gotVersion, anthropicVersion)
	}
	if gotContentType != "application/json" {
		t.Errorf("content-type header = %q, want application/json", gotContentType)
	}
}

func TestAnthropicProvider_Review_MalformedModelJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := anthropicMessageResponse{
			Content: []anthropicContentBlock{{Type: "text", Text: "I refuse to answer in JSON today."}},
			Usage:   anthropicUsage{InputTokens: 10, OutputTokens: 5},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	p := &AnthropicProvider{APIKey: testAPIKey, BaseURL: srv.URL}
	adv, err := p.Review(context.Background(), ReviewInput{})
	if err != nil {
		t.Fatalf("Review() error = %v, want nil (must degrade, not error)", err)
	}
	if adv.Enabled {
		t.Fatalf("Advisory.Enabled = true, want false for unparseable model output")
	}
	if adv.Err == "" {
		t.Error("Advisory.Err is empty, want a description of the parse failure")
	}
	if strings.Contains(adv.Err, testAPIKey) {
		t.Errorf("Advisory.Err leaked the API key: %q", adv.Err)
	}
}

func TestAnthropicProvider_Review_RetriesOnceOn500ThenSucceeds(t *testing.T) {
	restore := anthropicRetryBackoff
	anthropicRetryBackoff = time.Millisecond
	defer func() { anthropicRetryBackoff = restore }()

	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			w.WriteHeader(http.StatusInternalServerError)
			w.Write([]byte(`{"type":"error","error":{"type":"api_error","message":"boom"}}`))
			return
		}
		resp := anthropicMessageResponse{
			Content: []anthropicContentBlock{{Type: "text", Text: `{"summary":"ok after retry","warnings":[],"confidence":"high"}`}},
			Usage:   anthropicUsage{InputTokens: 1, OutputTokens: 1},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	p := &AnthropicProvider{APIKey: testAPIKey, BaseURL: srv.URL}
	adv, err := p.Review(context.Background(), ReviewInput{})
	if err != nil {
		t.Fatalf("Review() error = %v, want nil", err)
	}
	if calls != 2 {
		t.Fatalf("server received %d calls, want exactly 2 (one retry)", calls)
	}
	if !adv.Enabled || adv.Summary != "ok after retry" {
		t.Errorf("Advisory = %+v, want a successful review from the retried call", adv)
	}
}

func TestAnthropicProvider_Review_DoesNotRetryOn400(t *testing.T) {
	restore := anthropicRetryBackoff
	anthropicRetryBackoff = time.Millisecond
	defer func() { anthropicRetryBackoff = restore }()

	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"type":"error","error":{"type":"invalid_request_error","message":"bad request"}}`))
	}))
	defer srv.Close()

	p := &AnthropicProvider{APIKey: testAPIKey, BaseURL: srv.URL}
	adv, err := p.Review(context.Background(), ReviewInput{})
	if err != nil {
		t.Fatalf("Review() error = %v, want nil", err)
	}
	if calls != 1 {
		t.Errorf("server received %d calls, want exactly 1 (no retry on 400)", calls)
	}
	if adv.Enabled {
		t.Error("Advisory.Enabled = true, want false for a 400 response")
	}
}

func TestAnthropicProvider_Review_TimeoutDegradesGracefully(t *testing.T) {
	restore := anthropicRetryBackoff
	anthropicRetryBackoff = time.Millisecond
	defer func() { anthropicRetryBackoff = restore }()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(200 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	p := &AnthropicProvider{APIKey: testAPIKey, BaseURL: srv.URL, Timeout: 20 * time.Millisecond}

	done := make(chan struct{})
	var adv Advisory
	var err error
	go func() {
		adv, err = p.Review(context.Background(), ReviewInput{})
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("Review() did not return within 5s of a 20ms timeout — it must degrade, not hang")
	}

	if err != nil {
		t.Fatalf("Review() error = %v, want nil (timeout must degrade to Advisory, not an error)", err)
	}
	if adv.Enabled {
		t.Error("Advisory.Enabled = true, want false after a timeout")
	}
	if adv.Err == "" {
		t.Error("Advisory.Err is empty, want a timeout description")
	}
}

func TestAnthropicProvider_Review_APIKeyNeverLeaksIntoAdvisoryOrError(t *testing.T) {
	// A server that echoes the request body (including any accidental
	// leakage) back as an error message, to catch a key that somehow ended
	// up inside the request payload rather than just the header.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"type":"error","error":{"type":"invalid_request_error","message":"no api key here, just a generic failure"}}`))
	}))
	defer srv.Close()

	p := &AnthropicProvider{APIKey: testAPIKey, BaseURL: srv.URL}
	adv, err := p.Review(context.Background(), ReviewInput{Mode: "paper"})
	if err != nil {
		t.Fatalf("Review() error = %v, want nil", err)
	}

	b, _ := json.Marshal(adv)
	if strings.Contains(string(b), testAPIKey) {
		t.Fatalf("marshaled Advisory contains the API key: %s", b)
	}
	if err != nil && strings.Contains(err.Error(), testAPIKey) {
		t.Fatalf("returned error contains the API key: %v", err)
	}
}

func TestExtractReviewJSON_TolerateSurroundingProse(t *testing.T) {
	raw := "Sure thing! Here is my review:\n" +
		`{"summary": "all good", "warnings": [], "confidence": "high"}` +
		"\nHope that helps!"
	review, err := extractReviewJSON(raw)
	if err != nil {
		t.Fatalf("extractReviewJSON() error = %v", err)
	}
	if review.Summary != "all good" || review.Confidence != "high" {
		t.Errorf("review = %+v, want summary=all good confidence=high", review)
	}
}

func TestExtractReviewJSON_NoJSONObject(t *testing.T) {
	if _, err := extractReviewJSON("no json here at all"); err == nil {
		t.Fatal("extractReviewJSON() error = nil, want an error for text with no JSON object")
	}
}
