//! Integration tests against a mocked upstream (`wiremock`), exercising the
//! two handlers exactly as `dispatch` calls them, plus the real Actix app
//! wiring for the liveness probe.
//!
//! The `>=400` upstream-error branch (which overwrites
//! `~/.local/state/codex-router-last-error.json` on the real machine) is
//! deliberately not covered here to keep this suite hermetic — it was
//! exercised manually against the live Z.ai endpoint during development.

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
    let body = to_bytes(response.into_body()).await.expect("body");
    let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(value["id"], "resp_1");
}
