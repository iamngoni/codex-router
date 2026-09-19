//! Raw pass-through for providers that already speak the Responses API
//! (DeepSeek, Muse) plus the default fall-through to OpenAI itself. Only
//! tool-schema sanitization touches the body here — no shape translation.

use crate::config::{OPENAI_HOST, last_error_file};
use crate::keyfile::load_key;
use crate::logging::log;
use crate::routes::Route;
use crate::schema::sanitize_tools_for_route;
use crate::status::from_u16;
use actix_web::{HttpRequest, HttpResponse};
use serde_json::{Value, json};

const HOP_BY_HOP: [&str; 7] = [
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
];

/// Gives every server-sent `data:` frame the `event: <type>` line OpenAI and
/// DeepSeek put in front of theirs, for providers that omit it (OpenRouter).
/// The type comes from the payload's own `type` field, so nothing is invented;
/// a frame that has no `type` — keep-alive comments, `data: [DONE]` — is passed
/// through byte for byte, as is a stream that is already labelled.
fn label_sse_events(body: &[u8]) -> Vec<u8> {
    let text = String::from_utf8_lossy(body);
    let mut out = String::with_capacity(text.len() + text.len() / 64);
    let mut labelled = false;
    for line in text.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            labelled = false;
        } else if trimmed.starts_with("event: ") {
            labelled = true;
        } else if let Some(payload) = trimmed.strip_prefix("data: ") {
            let event = serde_json::from_str::<Value>(payload)
                .ok()
                .and_then(|value| value.get("type")?.as_str().map(str::to_string));
            if let Some(event) = event
                && !labelled
            {
                out.push_str("event: ");
                out.push_str(&event);
                out.push('\n');
            }
            labelled = true;
        }
        out.push_str(line);
    }
    out.into_bytes()
}

/// Forwards one request to `route`'s upstream unmodified (beyond tool-schema
/// sanitization and stripping hop-by-hop/auth headers), or to
/// `config::OPENAI_HOST` when `route` is `None`. Streams the upstream body
/// straight back and, for a routed 4xx/5xx, mirrors it to
/// [`last_error_file`] for quick debugging.
pub async fn handle_passthrough(
    client: &reqwest::Client,
    req: &HttpRequest,
    route: Option<&Route>,
    mut parsed: Option<Value>,
    raw_body: Vec<u8>,
) -> HttpResponse {
    let body: Vec<u8> = if let (Some(_), Some(parsed_val)) = (route, parsed.as_mut()) {
        sanitize_tools_for_route(parsed_val);
        serde_json::to_vec(parsed_val).unwrap_or(raw_body)
    } else {
        raw_body
    };

    let url = match route {
        Some(r) => format!("{}{}", r.base_url, r.path),
        None => format!(
            "https://{}{}",
            OPENAI_HOST,
            req.uri()
                .path_and_query()
                .map(|p| p.as_str().to_string())
                .unwrap_or_else(|| "/".to_string())
        ),
    };

    let method = reqwest::Method::from_bytes(req.method().as_str().as_bytes())
        .unwrap_or(reqwest::Method::POST);
    let mut builder = client.request(method, url);

    for (name, value) in req.headers() {
        let lower = name.as_str().to_lowercase();
        // `host` is deliberately dropped, not forwarded: reqwest derives
        // the correct Host header from the request URL above, which is the
        // upstream's host, not whatever Codex sent this proxy.
        if HOP_BY_HOP.contains(&lower.as_str())
            || lower == "host"
            || lower == "content-length"
            // The request body may already be decompressed by Actix.
            || lower == "content-encoding"
        {
            continue;
        }
        if route.is_some()
            && (lower.starts_with("authorization")
                || lower.starts_with("chatgpt-")
                || lower.starts_with("session_")
                || lower == "cookie")
        {
            continue;
        }
        if let Ok(v) = value.to_str() {
            builder = builder.header(name.as_str(), v);
        }
    }

    if let Some(r) = route {
        match load_key(&r.key_file) {
            Ok(key) => builder = builder.bearer_auth(key),
            Err(e) => {
                log(&format!("error key {:?}: {e}", r.key_file));
                return HttpResponse::InternalServerError()
                    .content_type("text/plain")
                    .body(format!("codex-router: cannot read {} key", r.name));
            }
        }
    }

    let model = parsed
        .as_ref()
        .and_then(|p| p.get("model"))
        .and_then(Value::as_str)
        .unwrap_or("-")
        .to_string();
    let route_name = route.map(|r| r.name).unwrap_or("openai");

    let sent = builder.body(body).send().await;
    let upstream = match sent {
        Ok(r) => r,
        Err(e) => {
            log(&format!(
                "error {route_name} {} {}: {e}",
                req.method(),
                req.uri()
            ));
            return HttpResponse::BadGateway()
                .content_type("text/plain")
                .body("codex-router: upstream error");
        }
    };

    let status = upstream.status();
    log(&format!(
        "{route_name} {} {} model={model} -> {}",
        req.method(),
        req.uri(),
        status.as_u16()
    ));

    let mut response_builder = HttpResponse::build(from_u16(status.as_u16()));
    for (name, value) in upstream.headers() {
        let lower = name.as_str().to_lowercase();
        if HOP_BY_HOP.contains(&lower.as_str()) {
            continue;
        }
        if let Ok(v) = value.to_str() {
            response_builder.append_header((name.as_str(), v));
        }
    }

    let mut bytes = upstream.bytes().await.unwrap_or_default();
    if route.is_some_and(|r| r.label_sse_events) {
        bytes = label_sse_events(&bytes).into();
    }

    if route.is_some() && status.as_u16() >= 400 {
        let text = String::from_utf8_lossy(&bytes).to_string();
        let _ = std::fs::write(
            last_error_file(),
            serde_json::to_vec_pretty(&json!({
                "route": route_name, "model": model, "url": req.uri().to_string(),
                "status": status.as_u16(), "body": text,
            }))
            .unwrap_or_default(),
        );
    }

    response_builder.body(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_data_frames_get_their_event_name_back() {
        let stream = ": \n\ndata: {\"type\":\"response.created\",\"response\":{}}\n\n\
                      data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n\
                      data: {\"type\":\"response.completed\",\"response\":{}}\n\n";
        let labelled = String::from_utf8(label_sse_events(stream.as_bytes())).expect("utf8");

        assert!(labelled.contains("event: response.created\ndata: {\"type\":\"response.created\""));
        assert!(labelled.contains("event: response.output_text.delta\n"));
        assert!(labelled.contains("event: response.completed\n"));
        assert!(labelled.starts_with(": \n"), "keep-alive comments survive");
    }

    #[test]
    fn a_stream_that_already_has_labels_is_untouched() {
        let stream = "event: response.created\ndata: {\"type\":\"response.created\"}\n\n";
        assert_eq!(label_sse_events(stream.as_bytes()), stream.as_bytes());
    }

    #[test]
    fn frames_without_a_type_are_passed_through() {
        let stream = "data: [DONE]\n\ndata: not json\n\n";
        assert_eq!(label_sse_events(stream.as_bytes()), stream.as_bytes());
    }

    #[test]
    fn a_non_sse_body_is_untouched() {
        let body = br#"{"id":"resp_1","output":[]}"#;
        assert_eq!(label_sse_events(body), body);
    }
}
