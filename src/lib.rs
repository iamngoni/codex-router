//! Library surface for `codex-router`, split out from `main.rs` so
//! integration tests can build the exact same Actix `App` the real binary
//! runs, instead of re-deriving the wiring by hand.

pub mod config;
pub mod keyfile;
pub mod logging;
pub mod proxy;
pub mod routes;
pub mod schema;
pub mod status;
pub mod translate;

use actix_web::{HttpRequest, HttpResponse, web};
use routes::{Translate, find_route};
use serde_json::Value;

/// Liveness probe used by the LaunchAgent/operator, and by nothing else —
/// Codex never calls this path.
pub async fn healthz() -> HttpResponse {
    HttpResponse::Ok().content_type("text/plain").body("ok")
}

/// Single entry point for every Codex request. Parses just enough of the
/// body (`model`) to pick a route, then hands off to the translating or
/// pass-through handler — or, for an unrecognized model (OpenAI's own),
/// falls through to the default pass-through with `route: None`.
pub async fn dispatch(
    req: HttpRequest,
    body: web::Bytes,
    client: web::Data<reqwest::Client>,
) -> HttpResponse {
    let raw = body.to_vec();
    let parsed: Option<Value> = serde_json::from_slice(&raw).ok();
    let model = parsed
        .as_ref()
        .and_then(|p| p.get("model"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let route = find_route(&model);

    if let (Some(r), Some(p)) = (&route, &parsed)
        && r.translate == Translate::Chat
    {
        return translate::handle_translated(client.get_ref(), r, p.clone()).await;
    }

    proxy::handle_passthrough(client.get_ref(), &req, route.as_ref(), parsed, raw).await
}
