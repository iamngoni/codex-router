//! Opt-in, real-network verification of the `openrouter/` route through
//! `dispatch`'s full routing glue — the path the hermetic suite can't reach,
//! because `find_route` always resolves to real provider hosts. It covers the
//! two things that are specific to this route: the Codex-side prefix is
//! stripped before forwarding, and OpenRouter's Responses endpoint really does
//! answer with the event types Codex expects when it streams.
//!
//! Requires a real `~/.config/openrouter/key` and spends a fraction of a cent
//! on a nano-class model, so it's `#[ignore]`d. The streaming test also pins the
//! `label_sse_events` fixup: OpenRouter sends bare `data:` frames, and Codex
//! expects the `event: <type>` label its other providers send.
//!
//! ```sh
//! cargo test --test live_openrouter -- --ignored
//! ```

use actix_web::test::TestRequest;
use actix_web::{App, web};
use serde_json::json;

const MODEL: &str = "openrouter/openai/gpt-5-nano";

/// The real app wiring from `main.rs`, in-process — like `live_glm.rs`.
macro_rules! real_app {
    () => {
        actix_web::test::init_service(
            App::new()
                .app_data(web::Data::new(reqwest::Client::new()))
                .app_data(web::PayloadConfig::new(
                    codex_router::config::MAX_PAYLOAD_BYTES,
                ))
                .default_service(web::route().to(codex_router::dispatch)),
        )
        .await
    };
}

fn ask(stream: bool) -> serde_json::Value {
    json!({
        "model": MODEL,
        "instructions": "You are a helpful assistant.",
        "input": [{ "type": "message", "role": "user", "content": [
            { "type": "input_text", "text": "Reply with exactly: ok" }
        ] }],
        "stream": stream,
    })
}

#[actix_web::test]
#[ignore]
async fn openrouter_answers_a_non_streaming_request() {
    let app = real_app!();
    let req = TestRequest::post()
        .uri("/backend-api/codex/responses")
        .set_json(ask(false))
        .to_request();
    let resp = actix_web::test::call_service(&app, req).await;

    let status = resp.status();
    let value: serde_json::Value = actix_web::test::read_body_json(resp).await;
    assert!(status.is_success(), "status={status:?} body={value}");
    // gpt-5-nano answers with a reasoning item before the message, so look for
    // the message rather than assuming it is first.
    let text = value["output"]
        .as_array()
        .expect("output")
        .iter()
        .filter(|item| item["type"] == "message")
        .filter_map(|item| item["content"][0]["text"].as_str())
        .collect::<String>();
    assert_eq!(text, "ok", "{value}");
    assert!(
        value["model"].as_str().unwrap_or_default() == "openai/gpt-5-nano",
        "the reply names the model OpenRouter ran, with the codex-side prefix gone: {value}"
    );
}

#[actix_web::test]
#[ignore]
async fn openrouter_streams_the_event_types_codex_expects() {
    let app = real_app!();
    let req = TestRequest::post()
        .uri("/backend-api/codex/responses")
        .set_json(ask(true))
        .to_request();
    let resp = actix_web::test::call_service(&app, req).await;
    assert!(resp.status().is_success(), "status={:?}", resp.status());

    let body = actix_web::test::read_body(resp).await;
    let text = String::from_utf8_lossy(&body);
    for expected in [
        "event: response.created",
        "event: response.output_text.delta",
        "event: response.completed",
    ] {
        assert!(text.contains(expected), "missing {expected} in:\n{text}");
    }
}
