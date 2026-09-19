//! Opt-in live checks for the native Anthropic Messages provider routes.
//!
//! These tests spend provider credits and require the configured key files;
//! they are ignored in the normal suite. Each request uses the same explicit
//! alias wiring as `main.rs`, including a query string on one compatibility
//! path.

use actix_web::test::TestRequest;
use actix_web::{App, web};
use serde_json::json;

macro_rules! messages_app {
    () => {
        actix_web::test::init_service(
            App::new()
                .app_data(web::Data::new(reqwest::Client::new()))
                .app_data(web::PayloadConfig::new(
                    codex_router::config::MAX_PAYLOAD_BYTES,
                ))
                .configure(codex_router::messages::configure),
        )
        .await
    };
}

fn body(model: &str) -> serde_json::Value {
    json!({
        "model": model,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
        "stream": false
    })
}

#[actix_web::test]
#[ignore]
async fn deepseek_messages_route_is_live() {
    let app = messages_app!();
    let request = TestRequest::post()
        .uri("/backend-api/claude")
        .insert_header(("authorization", "Bearer caller-is-replaced"))
        .insert_header(("cookie", "session=not-forwarded"))
        .set_json(body("deepseek-flash"))
        .to_request();
    let response = actix_web::test::call_service(&app, request).await;
    assert!(
        response.status().is_success(),
        "status={:?}",
        response.status()
    );
}

#[actix_web::test]
#[ignore]
async fn glm_messages_route_is_live() {
    let app = messages_app!();
    let request = TestRequest::post()
        .uri("/backend-api/claude/v1/messages")
        .set_json(body("glm-5.3-flash"))
        .to_request();
    let response = actix_web::test::call_service(&app, request).await;
    assert!(
        response.status().is_success(),
        "status={:?}",
        response.status()
    );
}

#[actix_web::test]
#[ignore]
async fn openrouter_messages_route_is_live() {
    let app = messages_app!();
    let request = TestRequest::post()
        .uri("/v1/messages?beta=true")
        .set_json(body("openrouter/openai/gpt-5-nano"))
        .to_request();
    let response = actix_web::test::call_service(&app, request).await;
    assert!(
        response.status().is_success(),
        "status={:?}",
        response.status()
    );
}
