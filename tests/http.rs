//! Integration tests against a mocked upstream (`wiremock`), exercising the
//! two handlers exactly as `dispatch` calls them, plus the real Actix app
//! wiring for the liveness probe.
//!
//! The upstream-error tests below intentionally overwrite the real
//! `~/.local/state/codex-router-last-error.json` on the machine running
//! `cargo test` (there is no test-only override for that path) — it is a
//! disposable "most recent error" scratch file the running service already
//! overwrites on every real failure, so a test run leaving synthetic data
//! there is a deliberately accepted, low-impact trade-off in exchange for
//! actually covering that write path.

use actix_web::body::to_bytes;
use actix_web::http::StatusCode;
use actix_web::test::TestRequest;
use actix_web::{App, web};
use codex_router::routes::{Route, Translate};
use serde_json::json;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Both upstream-error tests below write the same real
/// `last_error_file()` path — nothing test-local overrides it — so they'd
/// race under Rust's default parallel test execution without this. Async
/// (not `std::sync::Mutex`) because the guard is held across `.await`.
static LAST_ERROR_FILE_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn write_temp_key(contents: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let file = std::env::temp_dir().join(format!(
        "codex-router-test-key-{}-{nanos}",
        std::process::id()
    ));
    std::fs::write(&file, contents).expect("write temp key file");
    file
}

#[actix_web::test]
async fn healthz_reports_ok() {
    let app = actix_web::test::init_service(
        App::new().route("/healthz", web::get().to(codex_router::healthz)),
    )
    .await;
    let req = TestRequest::get().uri("/healthz").to_request();
    let resp = actix_web::test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::OK);
}

/// Regression test for a real incident: Actix's `web::Bytes` extractor
/// defaults to a 256 KiB body limit, which silently 413'd real Codex
/// requests (full conversation history + tool schemas routinely exceed
/// that) before `dispatch` ever ran — invisible in our own logs since the
/// handler never got called. Exercises the exact extractor `dispatch` uses,
/// with the exact `PayloadConfig` `main.rs` registers, without needing a
/// real provider round trip.
#[actix_web::test]
async fn large_body_is_not_rejected_by_the_payload_limit() {
    async fn echo_len(body: web::Bytes) -> actix_web::HttpResponse {
        actix_web::HttpResponse::Ok().body(body.len().to_string())
    }

    let app = actix_web::test::init_service(
        App::new()
            .app_data(web::PayloadConfig::new(
                codex_router::config::MAX_PAYLOAD_BYTES,
            ))
            .route("/echo", web::post().to(echo_len)),
    )
    .await;

    // Comfortably past Actix's 256 KiB default, comfortably under our limit.
    let big_body = vec![b'a'; 2 * 1024 * 1024];
    let req = TestRequest::post()
        .uri("/echo")
        .set_payload(big_body.clone())
        .to_request();
    let resp = actix_web::test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[actix_web::test]
async fn translated_route_returns_plain_text_reply() {
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/paas/v4/chat/completions"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-1",
            "choices": [{ "message": { "role": "assistant", "content": "ok" } }],
            "usage": { "prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4 },
        })))
        .mount(&mock_server)
        .await;

    let route = Route {
        name: "glm",
        prefix: "glm-",
        base_url: mock_server.uri(),
        path: "/api/paas/v4/chat/completions",
        key_file: write_temp_key("test-key"),
        translate: Translate::Chat,
        max_body_bytes: None,
        strip_prefix: false,
        label_sse_events: false,
    };

    let parsed = json!({
        "model": "glm-5.3-flash",
        "instructions": "You are helpful.",
        "input": [{ "type": "message", "role": "user", "content": [{ "type": "input_text", "text": "hi" }] }],
        "stream": false,
    });

    let client = reqwest::Client::new();
    let response = codex_router::translate::handle_translated(&client, &route, parsed).await;
    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body()).await.expect("body");
    let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(value["output"][0]["type"], "message");
    assert_eq!(value["output"][0]["content"][0]["text"], "ok");
    assert_eq!(value["usage"]["input_tokens"], 3);
}

#[actix_web::test]
async fn translated_route_returns_streaming_reply() {
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-2",
            "choices": [{ "message": { "role": "assistant", "content": "streamed" } }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 },
        })))
        .mount(&mock_server)
        .await;

    let route = Route {
        name: "glm",
        prefix: "glm-",
        base_url: mock_server.uri(),
        path: "/api/paas/v4/chat/completions",
        key_file: write_temp_key("test-key"),
        translate: Translate::Chat,
        max_body_bytes: None,
        strip_prefix: false,
        label_sse_events: false,
    };

    let parsed = json!({
        "model": "glm-5.3-flash",
        "input": [{ "type": "message", "role": "user", "content": [{ "type": "input_text", "text": "hi" }] }],
        "stream": true,
    });

    let client = reqwest::Client::new();
    let response = codex_router::translate::handle_translated(&client, &route, parsed).await;
    assert_eq!(response.status(), StatusCode::OK);

    let body = to_bytes(response.into_body()).await.expect("body");
    let text = String::from_utf8(body.to_vec()).expect("utf8");
    assert!(text.contains("event: response.created"));
    assert!(text.contains("\"delta\":\"streamed\""));
    assert!(text.contains("event: response.completed"));
}

#[actix_web::test]
async fn translated_route_returns_tool_call() {
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-3",
            "choices": [{ "message": { "tool_calls": [
                { "id": "call_1", "type": "function", "function": { "name": "get_weather", "arguments": "{\"city\":\"Harare\"}" } }
            ] } }],
            "usage": { "prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7 },
        })))
        .mount(&mock_server)
        .await;

    let route = Route {
        name: "glm",
        prefix: "glm-",
        base_url: mock_server.uri(),
        path: "/api/paas/v4/chat/completions",
        key_file: write_temp_key("test-key"),
        translate: Translate::Chat,
        max_body_bytes: None,
        strip_prefix: false,
        label_sse_events: false,
    };

    let parsed = json!({
        "model": "glm-5.3",
        "input": [{ "type": "message", "role": "user", "content": [{ "type": "input_text", "text": "weather in Harare?" }] }],
        "tools": [{ "type": "function", "name": "get_weather", "parameters": { "type": "object" } }],
        "stream": false,
    });

    let client = reqwest::Client::new();
    let response = codex_router::translate::handle_translated(&client, &route, parsed).await;
    let body = to_bytes(response.into_body()).await.expect("body");
    let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(value["output"][0]["type"], "function_call");
    assert_eq!(value["output"][0]["call_id"], "call_1");
    assert_eq!(value["output"][0]["name"], "get_weather");
}

#[actix_web::test]
async fn passthrough_route_replaces_client_auth_with_route_key() {
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(header("authorization", "Bearer upstream-key"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-encoding", "gzip")
                .set_body_json(json!({ "id": "resp_1", "object": "response" })),
        )
        .mount(&mock_server)
        .await;

    let route = Route {
        name: "deepseek",
        prefix: "deepseek-",
        base_url: mock_server.uri(),
        path: "/responses",
        key_file: write_temp_key("upstream-key"),
        translate: Translate::None,
        max_body_bytes: None,
        strip_prefix: false,
        label_sse_events: false,
    };

    let raw_body = json!({ "model": "deepseek-flash", "input": [] })
        .to_string()
        .into_bytes();
    let req = TestRequest::post()
        .uri("/backend-api/codex/responses")
        .insert_header((
            "authorization",
            "Bearer client-side-token-should-not-reach-upstream",
        ))
        .insert_header(("content-encoding", "gzip"))
        .insert_header(("cookie", "session=should-not-reach-upstream"))
        .to_http_request();

    let client = reqwest::Client::new();
    let parsed: serde_json::Value = serde_json::from_slice(&raw_body).unwrap();
    let response = codex_router::proxy::handle_passthrough(
        &client,
        &req,
        Some(&route),
        Some(parsed),
        raw_body,
    )
    .await;

    // wiremock's `header` matcher on the mount above already asserts the
    // outgoing request carried the route's key, not the client's; a 200
    // here proves that matched (a mismatch would 404 on the unmounted path).
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("content-encoding")
            .and_then(|value| value.to_str().ok()),
        Some("gzip")
    );
    let body = to_bytes(response.into_body()).await.expect("body");
    let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(value["id"], "resp_1");
    let received = mock_server
        .received_requests()
        .await
        .expect("recorded request");
    let headers = &received[0].headers;
    assert_eq!(
        headers
            .get("authorization")
            .and_then(|value| value.to_str().ok()),
        Some("Bearer upstream-key")
    );
    assert!(!headers.contains_key("content-encoding"));
    assert!(!headers.contains_key("cookie"));
}

#[actix_web::test]
async fn passthrough_route_missing_key_file_returns_500() {
    let route = Route {
        name: "deepseek",
        prefix: "deepseek-",
        base_url: "https://example.invalid".to_string(),
        path: "/responses",
        key_file: std::env::temp_dir().join("codex-router-test-key-does-not-exist"),
        translate: Translate::None,
        max_body_bytes: None,
        strip_prefix: false,
        label_sse_events: false,
    };
    let req = TestRequest::post().uri("/x").to_http_request();
    let client = reqwest::Client::new();
    let response =
        codex_router::proxy::handle_passthrough(&client, &req, Some(&route), None, Vec::new())
            .await;
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
}

#[actix_web::test]
async fn passthrough_route_connect_failure_returns_502() {
    // Port 1 is a privileged port nothing binds to in this environment —
    // connection is refused immediately, no real network round trip.
    let route = Route {
        name: "deepseek",
        prefix: "deepseek-",
        base_url: "http://127.0.0.1:1".to_string(),
        path: "/responses",
        key_file: write_temp_key("k"),
        translate: Translate::None,
        max_body_bytes: None,
        strip_prefix: false,
        label_sse_events: false,
    };
    let req = TestRequest::post().uri("/x").to_http_request();
    let client = reqwest::Client::new();
    let response =
        codex_router::proxy::handle_passthrough(&client, &req, Some(&route), None, Vec::new())
            .await;
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
}

#[actix_web::test]
async fn passthrough_route_upstream_error_is_forwarded_and_recorded() {
    let _guard = LAST_ERROR_FILE_LOCK.lock().await;
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(402).set_body_json(json!({ "error": "insufficient balance" })),
        )
        .mount(&mock_server)
        .await;

    let route = Route {
        name: "deepseek",
        prefix: "deepseek-",
        base_url: mock_server.uri(),
        path: "/responses",
        key_file: write_temp_key("k"),
        translate: Translate::None,
        max_body_bytes: None,
        strip_prefix: false,
        label_sse_events: false,
    };
    let raw_body = json!({ "model": "deepseek-flash" })
        .to_string()
        .into_bytes();
    let req = TestRequest::post().uri("/x").to_http_request();
    let client = reqwest::Client::new();
    let parsed: serde_json::Value = serde_json::from_slice(&raw_body).unwrap();
    let response = codex_router::proxy::handle_passthrough(
        &client,
        &req,
        Some(&route),
        Some(parsed),
        raw_body,
    )
    .await;
    assert_eq!(response.status(), StatusCode::PAYMENT_REQUIRED);

    // The >=400 branch overwrites the real last-error file — read it back
    // to confirm this specific run's error was recorded, rather than
    // asserting nothing about the known side effect.
    let recorded: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(codex_router::config::last_error_file()).unwrap(),
    )
    .unwrap();
    assert_eq!(recorded["route"], "deepseek");
    assert_eq!(recorded["status"], 402);
}

#[actix_web::test]
async fn translated_route_missing_key_file_returns_500() {
    let route = Route {
        name: "glm",
        prefix: "glm-",
        base_url: "https://example.invalid".to_string(),
        path: "/api/paas/v4/chat/completions",
        key_file: std::env::temp_dir().join("codex-router-test-key-does-not-exist"),
        translate: Translate::Chat,
        max_body_bytes: None,
        strip_prefix: false,
        label_sse_events: false,
    };
    let parsed = json!({ "model": "glm-5.3-flash", "input": [] });
    let client = reqwest::Client::new();
    let response = codex_router::translate::handle_translated(&client, &route, parsed).await;
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
}

#[actix_web::test]
async fn translated_route_connect_failure_returns_502() {
    let route = Route {
        name: "glm",
        prefix: "glm-",
        base_url: "http://127.0.0.1:1".to_string(),
        path: "/api/paas/v4/chat/completions",
        key_file: write_temp_key("k"),
        translate: Translate::Chat,
        max_body_bytes: None,
        strip_prefix: false,
        label_sse_events: false,
    };
    let parsed = json!({ "model": "glm-5.3-flash", "input": [] });
    let client = reqwest::Client::new();
    let response = codex_router::translate::handle_translated(&client, &route, parsed).await;
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
}

#[actix_web::test]
async fn translated_route_upstream_error_is_forwarded_and_recorded() {
    let _guard = LAST_ERROR_FILE_LOCK.lock().await;
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(429).set_body_json(json!({ "error": "overloaded" })))
        .mount(&mock_server)
        .await;

    let route = Route {
        name: "glm",
        prefix: "glm-",
        base_url: mock_server.uri(),
        path: "/api/paas/v4/chat/completions",
        key_file: write_temp_key("k"),
        translate: Translate::Chat,
        max_body_bytes: None,
        strip_prefix: false,
        label_sse_events: false,
    };
    let parsed = json!({ "model": "glm-4.7-flash", "input": [] });
    let client = reqwest::Client::new();
    let response = codex_router::translate::handle_translated(&client, &route, parsed).await;
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);

    let recorded: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(codex_router::config::last_error_file()).unwrap(),
    )
    .unwrap();
    assert_eq!(recorded["route"], "glm");
    assert_eq!(recorded["status"], 429);
}

#[actix_web::test]
async fn translated_route_malformed_upstream_json_returns_502() {
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
        .mount(&mock_server)
        .await;

    let route = Route {
        name: "glm",
        prefix: "glm-",
        base_url: mock_server.uri(),
        path: "/api/paas/v4/chat/completions",
        key_file: write_temp_key("k"),
        translate: Translate::Chat,
        max_body_bytes: None,
        strip_prefix: false,
        label_sse_events: false,
    };
    let parsed = json!({ "model": "glm-5.3-flash", "input": [] });
    let client = reqwest::Client::new();
    let response = codex_router::translate::handle_translated(&client, &route, parsed).await;
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
}

/// OpenRouter streams Responses events as bare `data:` frames; Codex expects the
/// `event: <type>` label every other provider it talks to sends, so a route that
/// declares the difference gets it added on the way back — and one that does not
/// declare it is passed through byte for byte.
#[actix_web::test]
async fn sse_frames_are_labelled_only_for_routes_that_ask() {
    let mock_server = MockServer::start().await;
    let stream =
        "data: {\"type\":\"response.created\"}\n\ndata: {\"type\":\"response.completed\"}\n\n";
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(stream),
        )
        .mount(&mock_server)
        .await;

    let route = |label: bool| Route {
        name: "openrouter",
        prefix: "openrouter/",
        base_url: mock_server.uri(),
        path: "/api/v1/responses",
        key_file: write_temp_key("k"),
        translate: Translate::None,
        max_body_bytes: None,
        strip_prefix: true,
        label_sse_events: label,
    };
    let raw = json!({ "model": "anthropic/claude-x", "input": [] })
        .to_string()
        .into_bytes();
    let client = reqwest::Client::new();

    for (label, expected) in [(true, true), (false, false)] {
        let route = route(label);
        let req = TestRequest::post().uri("/x").to_http_request();
        let response =
            codex_router::proxy::handle_passthrough(&client, &req, Some(&route), None, raw.clone())
                .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body()).await.expect("body");
        let text = String::from_utf8_lossy(&body);
        assert_eq!(
            text.contains("event: response.created\n"),
            expected,
            "label_sse_events={label}: {text}"
        );
        assert_eq!(
            text.contains("event: response.completed\n"),
            expected,
            "{text}"
        );
    }
}

/// A body past the route's measured upstream limit is refused here, not
/// uploaded: the upstream answers those with a bare HTML 413 anyway, and Codex
/// retries a few times per turn. Posts through the real app wiring (`main.rs`'s
/// `PayloadConfig` + `default_service`), so it also proves a 48 MiB body clears
/// the extractor, and pins the limit `dispatch` enforces to the route's own.
#[actix_web::test]
async fn deepseek_body_over_the_edge_limit_is_refused_locally() {
    let cap = codex_router::routes::DEEPSEEK_MAX_BODY_BYTES;
    let prefix = br#"{"model":"deepseek-flash","input":""#;
    let suffix = br#""}"#;
    let mut payload = Vec::with_capacity(cap + 1);
    payload.extend_from_slice(prefix);
    payload.resize(cap + 1 - suffix.len(), b'x');
    payload.extend_from_slice(suffix);
    assert_eq!(payload.len(), cap + 1);

    let app = actix_web::test::init_service(
        App::new()
            .app_data(web::Data::new(reqwest::Client::new()))
            .app_data(web::PayloadConfig::new(
                codex_router::config::MAX_PAYLOAD_BYTES,
            ))
            .default_service(web::route().to(codex_router::dispatch)),
    )
    .await;
    let req = TestRequest::post()
        .uri("/backend-api/codex/responses")
        .set_payload(payload)
        .to_request();
    let response = actix_web::test::call_service(&app, req).await;

    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let body = to_bytes(response.into_body()).await.expect("body");
    let value: serde_json::Value = serde_json::from_slice(&body).expect("json error object");
    assert_eq!(value["error"]["code"], "payload_too_large");
    let message = value["error"]["message"].as_str().expect("message");
    assert!(message.contains("deepseek"), "{message}");
    assert!(message.contains("48.0 MiB"), "{message}");
}
