//! Opt-in, real-network verification of `dispatch`'s full routing glue —
//! the one path the hermetic `tests/http.rs` suite can't reach, since
//! `find_route` isn't test-injectable and always resolves to the real
//! provider hosts. Requires a real `~/.config/zai/key` and spends real
//! Z.ai credits, so it's `#[ignore]`d: run explicitly with
//! `cargo test --test live_glm -- --ignored`.

use actix_web::test::TestRequest;
use actix_web::{App, web};
use serde_json::json;

#[actix_web::test]
#[ignore]
async fn dispatch_routes_glm_model_through_the_real_app_wiring() {
    let client = reqwest::Client::new();
    let app = actix_web::test::init_service(
        App::new()
            .app_data(web::Data::new(client))
            .app_data(web::PayloadConfig::new(
                codex_router::config::MAX_PAYLOAD_BYTES,
            ))
            .route("/healthz", web::get().to(codex_router::healthz))
            .default_service(web::route().to(codex_router::dispatch)),
    )
    .await;

    let body = json!({
        "model": "glm-5.3-flash",
        "instructions": "You are a helpful assistant.",
        "input": [{ "type": "message", "role": "user", "content": [{ "type": "input_text", "text": "Reply with exactly: ok" }] }],
        "stream": false,
    });

    let req = TestRequest::post()
        .uri("/backend-api/codex/responses")
        .set_json(&body)
        .to_request();
    let resp = actix_web::test::call_service(&app, req).await;
    assert!(resp.status().is_success(), "status={:?}", resp.status());

    let value: serde_json::Value = actix_web::test::read_body_json(resp).await;
    assert_eq!(value["output"][0]["type"], "message");
}
