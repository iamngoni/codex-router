//! Hermetic route-wiring checks for the native Anthropic Messages namespace.
//!
//! These tests use the same `messages::configure` table as the binary, then
//! retain the normal dispatcher as a fallback to prove reserved paths never
//! reach it.

use actix_web::body::to_bytes;
use actix_web::http::StatusCode;
use actix_web::test::TestRequest;
use actix_web::{App, web};
use codex_router::messages::ROUTE_SPECS;
use serde_json::Value;

#[actix_web::test]
async fn every_shared_post_alias_reaches_messages_validation() {
    let app = actix_web::test::init_service(
        App::new()
            .app_data(web::Data::new(reqwest::Client::new()))
            .configure(codex_router::messages::configure)
            .default_service(web::route().to(codex_router::dispatch)),
    )
    .await;

    for spec in ROUTE_SPECS {
        let request = TestRequest::post()
            .uri(spec.path)
            .set_payload(b"{".as_slice())
            .to_request();
        let response = actix_web::test::call_service(&app, request).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{}", spec.path);
        let body: Value =
            serde_json::from_slice(&to_bytes(response.into_body()).await.expect("body"))
                .expect("Anthropic error JSON");
        assert_eq!(body["type"], "error", "{}", spec.path);
        assert_eq!(
            body["error"]["type"], "invalid_request_error",
            "{}",
            spec.path
        );
    }
}

#[actix_web::test]
async fn unmatched_namespace_paths_and_methods_are_local_not_openai_fallbacks() {
    let app = actix_web::test::init_service(
        App::new()
            .app_data(web::Data::new(reqwest::Client::new()))
            .configure(codex_router::messages::configure)
            .default_service(web::route().to(codex_router::dispatch)),
    )
    .await;

    for path in [
        "/backend-api/claude/v1/not-a-messages-route",
        "/v1/messages/not-a-messages-route",
        "/backend-api/claudefuture",
    ] {
        let request = TestRequest::post()
            .uri(path)
            .insert_header(("authorization", "Bearer caller-secret"))
            .set_payload(r#"{"model":"gpt-6-astra","messages":[]}"#)
            .to_request();
        let response = actix_web::test::call_service(&app, request).await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{path}");
        let body: Value =
            serde_json::from_slice(&to_bytes(response.into_body()).await.expect("body"))
                .expect("Anthropic error JSON");
        assert_eq!(body["type"], "error", "{path}");
        assert_eq!(body["error"]["type"], "not_found_error", "{path}");
    }

    let request = TestRequest::get()
        .uri(codex_router::messages::V1_PATH)
        .to_request();
    let response = actix_web::test::call_service(&app, request).await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}
