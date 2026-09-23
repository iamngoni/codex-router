//! Library surface for `codex-router`, split out from `main.rs` so
//! integration tests can build the exact same Actix `App` the real binary
//! runs, instead of re-deriving the wiring by hand.

pub mod config;
pub mod keyfile;
pub mod logging;
pub mod messages;
pub mod models;
pub mod proxy;
pub mod routes;
pub mod schema;
pub mod status;
pub mod translate;
pub mod trim;

use actix_web::{HttpRequest, HttpResponse, web};
use routes::{Route, Translate, find_route};
use serde_json::{Value, json};

/// Bytes of request body kept in the request-log line. Full conversations run
/// to tens of megabytes; a recognizable prefix plus the true total is what an
/// operator needs, and it keeps one request to one readable log line.
const MAX_LOGGED_BODY_BYTES: usize = 2048;

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
    if req.method() == actix_web::http::Method::GET && req.path() == "/backend-api/codex/models" {
        let state = req
            .app_data::<web::Data<models::ModelsState>>()
            .cloned()
            .unwrap_or_else(|| web::Data::new(models::ModelsState::new()));
        return models::handle(req, client, state).await;
    }

    // Explicit Messages routes are configured in `main`, but this guard also
    // protects suffixes/methods that Actix cannot match to those resources.
    // They must never fall through to the OpenAI/Codex dispatcher with a
    // caller's Anthropic credential attached.
    if messages::is_reserved_path(req.path()) {
        return messages::reject_namespace(req).await;
    }
    let raw = body.to_vec();
    let mut parsed: Option<Value> = serde_json::from_slice(&raw).ok();
    let model = parsed
        .as_ref()
        .and_then(|p| p.get("model"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let route = find_route(&model);
    let upstream_model = route
        .as_ref()
        .and_then(|r| rewrite_model_for_route(r, &model, &mut parsed));

    // Log every incoming request with its body (truncated to keep logs
    // manageable — full conversation histories can be megabytes).
    let route_name = route.as_ref().map(|r| r.name).unwrap_or("openai");
    let body_snapshot: String = match &parsed {
        Some(v) => body_preview(&serde_json::to_string(v).unwrap_or_default()),
        None => {
            let head = raw.len().min(MAX_LOGGED_BODY_BYTES);
            String::from_utf8_lossy(&raw[..head]).to_string()
        }
    };
    logging::log(&format!(
        ">> {} {} model={} route={}{} body={}",
        req.method(),
        req.uri(),
        if model.is_empty() { "-" } else { &model },
        route_name,
        match &upstream_model {
            Some(name) => format!(" forwarded={name}"),
            None => String::new(),
        },
        body_snapshot
    ));

    if let Some(r) = &route {
        match enforce_edge_limit(r, &mut parsed, raw.len()) {
            Enforced::Fits => {}
            Enforced::Trimmed(report) => logging::log(&format!(
                "trimmed route={} model={} {} (cap {})",
                r.name,
                model,
                report.summary(),
                r.max_body_bytes.map(trim::human_bytes).unwrap_or_default()
            )),
            Enforced::Refused(refusal) => {
                logging::log(&format!(
                    "refused route={} model={} body={} cap={}",
                    r.name,
                    model,
                    raw.len(),
                    r.max_body_bytes.map(trim::human_bytes).unwrap_or_default()
                ));
                return refusal;
            }
        }
    }

    if let (Some(r), Some(p)) = (&route, &parsed)
        && r.translate == Translate::Chat
    {
        return translate::handle_translated(client.get_ref(), r, p.clone()).await;
    }

    proxy::handle_passthrough(client.get_ref(), &req, route.as_ref(), parsed, raw).await
}

/// What enforcing a route's measured upstream limit came to.
enum Enforced {
    /// Nothing to do: no measured limit for this route, or the request fits.
    Fits,
    Trimmed(trim::TrimReport),
    Refused(HttpResponse),
}

/// Enforces a route's measured upstream limit: trim the request down to the cap
/// where [`trim`] can do that without touching the conversation itself, and
/// build the local refusal where it cannot.
fn enforce_edge_limit(route: &Route, parsed: &mut Option<Value>, body_len: usize) -> Enforced {
    let Some(cap) = exceeded_cap(route, body_len) else {
        return Enforced::Fits;
    };
    let Some(body) = parsed.as_mut() else {
        return Enforced::Refused(body_too_large(route, body_len, cap, None));
    };
    match trim::fit_to_cap(body, cap) {
        // Codex's bytes were an eyelash over the cap, but what we forward — the
        // body re-serialized from `parsed` — is not.
        trim::Fitted::Untouched => Enforced::Fits,
        trim::Fitted::Trimmed(report) => Enforced::Trimmed(report),
        // A trim that gave up may still have removed a lot first; the refusal
        // says what it tried, otherwise it reads as if nothing had been done.
        trim::Fitted::CannotFit(report) => {
            let left = trim::serialized_len(body);
            Enforced::Refused(body_too_large(route, body_len, cap, Some((&report, left))))
        }
    }
}

/// Points the body at the provider's own model name when the route's prefix is
/// Codex's naming rather than the provider's — `openrouter/anthropic/claude-x`
/// has to reach OpenRouter as `anthropic/claude-x`. Returns the name sent
/// upstream when it differs from the one Codex used, for the request log.
fn rewrite_model_for_route(
    route: &Route,
    model: &str,
    parsed: &mut Option<Value>,
) -> Option<String> {
    let upstream = route.upstream_model(model)?;
    if let Some(body) = parsed.as_mut() {
        body["model"] = Value::String(upstream.clone());
    }
    Some(upstream)
}

/// The route's edge limit, when `body_len` is past it. `None` covers both "no
/// measured limit for this route" and "this request fits" — dispatch only acts
/// on a body its upstream would have rejected anyway.
fn exceeded_cap(route: &Route, body_len: usize) -> Option<usize> {
    route.max_body_bytes.filter(|cap| body_len > *cap)
}

/// Builds the local 413 for a body the route's upstream would answer with its
/// own opaque error page. Shaped like a Responses API error object so Codex
/// shows the sizes and the cause instead of a bare "openresty" HTML page.
fn body_too_large(
    route: &Route,
    body_len: usize,
    cap: usize,
    trimmed: Option<(&trim::TrimReport, usize)>,
) -> HttpResponse {
    let trimmed = match trimmed {
        Some((report, left)) if report.bytes_saved > 0 => format!(
            ", and trimming {} only got it to {}",
            report.clause(),
            trim::human_bytes(left)
        ),
        _ => String::new(),
    };
    let message = format!(
        "codex-router: request body is {}{trimmed}, over {}'s {}. Its API edge \
         rejects that before the model sees it — shrink the conversation (an \
         oversized attachment is the usual cause) or send this turn to a \
         different model.",
        trim::human_bytes(body_len),
        route.name,
        trim::human_bytes(cap)
    );
    HttpResponse::PayloadTooLarge()
        .content_type("application/json")
        .json(json!({
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": "payload_too_large",
            }
        }))
}

/// Request-log preview of a serialized body: its first
/// [`MAX_LOGGED_BODY_BYTES`] bytes plus the real size. The cut is moved back to
/// a character boundary rather than slicing blindly — a body whose 2048th byte
/// lands inside a multi-byte character would otherwise panic the handler.
fn body_preview(body: &str) -> String {
    if body.len() <= MAX_LOGGED_BODY_BYTES {
        return body.to_string();
    }
    let mut cut = MAX_LOGGED_BODY_BYTES;
    while !body.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}…({} bytes total)", &body[..cut], body.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::body::to_bytes;
    use actix_web::http::StatusCode;
    use std::path::PathBuf;

    fn route_with_cap(cap: Option<usize>) -> Route {
        Route {
            name: "deepseek",
            prefix: "deepseek-",
            base_url: "https://api.deepseek.com".to_string(),
            path: "/responses",
            key_file: PathBuf::from("/nonexistent/key"),
            translate: Translate::None,
            max_body_bytes: cap,
            strip_prefix: false,
            label_sse_events: false,
        }
    }

    #[test]
    fn cap_is_only_exceeded_past_the_limit() {
        let capped = route_with_cap(Some(1024));
        assert_eq!(
            exceeded_cap(&capped, 1024),
            None,
            "a body exactly at the limit is the provider's call, not ours"
        );
        assert_eq!(exceeded_cap(&capped, 1025), Some(1024));

        // No measured limit means no local opinion, whatever the size.
        assert_eq!(exceeded_cap(&route_with_cap(None), usize::MAX), None);
    }

    #[test]
    fn body_preview_never_splits_a_character() {
        assert_eq!(body_preview("hello"), "hello");

        // Two-byte characters: byte 2048 is already a boundary.
        let two_byte = "ß".repeat(2000);
        assert!(body_preview(&two_byte).ends_with("…(4000 bytes total)"));

        // Three-byte characters: byte 2048 lands mid-character, so the cut
        // has to walk back — the plain `&body[..2048]` this replaced panicked
        // on exactly this input.
        let three_byte = "€".repeat(1000);
        let preview = body_preview(&three_byte);
        assert!(preview.starts_with(&"€".repeat(682)));
        assert!(preview.ends_with("…(3000 bytes total)"));
    }

    #[actix_web::test]
    async fn refusal_names_both_sizes() {
        let cap = routes::DEEPSEEK_MAX_BODY_BYTES;
        let resp = body_too_large(&route_with_cap(Some(cap)), 53_179_956, cap, None);
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);

        let bytes = to_bytes(resp.into_body()).await.expect("body");
        let body: Value = serde_json::from_slice(&bytes).expect("error object");
        let message = body["error"]["message"].as_str().expect("message");
        assert!(message.contains("50.7 MiB"), "{message}");
        assert!(message.contains("48.0 MiB"), "{message}");
        assert!(message.contains("deepseek"), "{message}");
    }

    #[test]
    fn routed_prefixes_are_stripped_from_the_forwarded_model() {
        let route = Route {
            name: "openrouter",
            prefix: "openrouter/",
            base_url: "https://openrouter.ai".to_string(),
            path: "/api/v1/responses",
            key_file: PathBuf::from("/nonexistent/key"),
            translate: Translate::None,
            max_body_bytes: None,
            strip_prefix: true,
            label_sse_events: true,
        };
        let mut parsed = Some(json!({ "model": "openrouter/anthropic/claude-sonnet-4.5" }));

        let forwarded = rewrite_model_for_route(
            &route,
            "openrouter/anthropic/claude-sonnet-4.5",
            &mut parsed,
        );

        assert_eq!(forwarded.as_deref(), Some("anthropic/claude-sonnet-4.5"));
        assert_eq!(
            parsed.as_ref().expect("body")["model"],
            "anthropic/claude-sonnet-4.5"
        );
    }

    #[test]
    fn routes_without_a_codex_side_prefix_leave_the_model_alone() {
        let mut parsed = Some(json!({ "model": "deepseek-flash" }));
        let forwarded =
            rewrite_model_for_route(&route_with_cap(None), "deepseek-flash", &mut parsed);
        assert_eq!(forwarded, None);
        assert_eq!(parsed.as_ref().expect("body")["model"], "deepseek-flash");
    }

    #[test]
    fn trimming_happens_for_a_request_over_the_cap() {
        let route = route_with_cap(Some(4_000));
        let mut parsed = Some(json!({ "input": [
            { "type": "function_call_output", "call_id": "1", "output": "x".repeat(20_000) }
        ] }));

        let Enforced::Trimmed(report) = enforce_edge_limit(&route, &mut parsed, 20_500) else {
            panic!("expected a trim");
        };

        assert!(report.bytes_saved > 0, "{report:?}");
        let body = parsed.as_ref().expect("body");
        assert!(trim::serialized_len(body) <= 4_000);
    }

    #[test]
    fn refusal_happens_when_nothing_can_absorb_the_cut() {
        let route = route_with_cap(Some(4_000));
        let mut parsed = Some(json!({ "input": [
            { "type": "message", "role": "user", "content": [
                { "type": "input_text", "text": "u".repeat(20_000) }
            ] }
        ] }));

        let Enforced::Refused(refusal) = enforce_edge_limit(&route, &mut parsed, 20_500) else {
            panic!("expected a refusal");
        };
        assert_eq!(refusal.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }
}
