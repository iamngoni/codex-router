//! Native Anthropic Messages proxy for Claude-compatible clients.
//!
//! This module owns request validation, provider selection, credential
//! isolation, and byte-preserving streaming for the three local Messages
//! paths. It deliberately does not translate to the Responses API: a valid
//! Messages body is forwarded unchanged unless the local `openrouter/` model
//! prefix must be removed.

use crate::keyfile::load_key;
use crate::logging::log;
use crate::routes::{Route, find_route};
use actix_web::http::StatusCode;
use actix_web::{HttpRequest, HttpResponse, web};
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::{Value, json};
use std::path::PathBuf;

/// Canonical logical endpoint used by the local Claude provider configuration.
pub const CANONICAL_PATH: &str = "/backend-api/claude";
/// Native Anthropic SDK compatibility endpoint.
pub const V1_PATH: &str = "/backend-api/claude/v1/messages";
/// Stock Claude Code endpoint when the base URL is set to this router root.
pub const ROOT_V1_PATH: &str = "/v1/messages";
/// Canonical Messages count-tokens endpoint.
pub const CANONICAL_COUNT_TOKENS_PATH: &str = "/backend-api/claude/count_tokens";
/// Claude SDK compatibility count-tokens endpoint.
pub const V1_COUNT_TOKENS_PATH: &str = "/backend-api/claude/v1/messages/count_tokens";
/// Root compatibility count-tokens endpoint.
pub const ROOT_V1_COUNT_TOKENS_PATH: &str = "/v1/messages/count_tokens";

/// Native operation selected by a local route specification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    /// Anthropic's ordinary message generation endpoint.
    Messages,
    /// Anthropic's token counting endpoint.
    CountTokens,
}

/// One locally-owned endpoint. Main wiring and hermetic route tests consume
/// this same table so an SDK alias cannot silently bypass the handler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouteSpec {
    /// Local path, excluding its query string.
    pub path: &'static str,
    /// Operation represented by the path.
    pub operation: Operation,
}

/// All supported POST aliases, in canonical-to-compatibility order.
pub const ROUTE_SPECS: [RouteSpec; 6] = [
    RouteSpec {
        path: CANONICAL_PATH,
        operation: Operation::Messages,
    },
    RouteSpec {
        path: V1_PATH,
        operation: Operation::Messages,
    },
    RouteSpec {
        path: ROOT_V1_PATH,
        operation: Operation::Messages,
    },
    RouteSpec {
        path: CANONICAL_COUNT_TOKENS_PATH,
        operation: Operation::CountTokens,
    },
    RouteSpec {
        path: V1_COUNT_TOKENS_PATH,
        operation: Operation::CountTokens,
    },
    RouteSpec {
        path: ROOT_V1_COUNT_TOKENS_PATH,
        operation: Operation::CountTokens,
    },
];

/// Returns whether a path belongs to the reserved Claude namespace. The
/// dispatch fallback checks this too, protecting malformed suffixes that do
/// not match an Actix resource pattern from ever reaching chatgpt.com.
pub fn is_reserved_path(path: &str) -> bool {
    path.starts_with("/backend-api/claude") || path.starts_with("/v1/messages")
}

/// Shared Actix wiring for every Messages alias and its local catch-all guard.
/// Unknown paths and methods return an Anthropic-shaped local 404 instead of
/// falling through to the OpenAI/Codex dispatcher.
pub fn configure(config: &mut web::ServiceConfig) {
    for spec in ROUTE_SPECS {
        config.service(
            web::resource(spec.path)
                .route(web::post().to(handle_messages))
                .route(web::to(reject_namespace)),
        );
    }
    config
        .service(web::resource("/backend-api/claude/{tail:.*}").route(web::to(reject_namespace)))
        .service(web::resource("/v1/messages/{tail:.*}").route(web::to(reject_namespace)));
}

const HOP_BY_HOP: [&str; 7] = [
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
];

#[derive(Clone)]
struct Target {
    route_name: &'static str,
    base_url: String,
    path: &'static str,
    forwarded_model: String,
    auth: TargetAuth,
}

#[derive(Clone)]
enum TargetAuth {
    /// Claude's first-party endpoint receives the caller's own credential.
    Claude,
    /// Every third-party provider receives only its configured route key.
    RouteKey(PathBuf),
}

/// Handles all three local Messages paths. The route registrations in `main`
/// intentionally point to this one handler so aliases cannot diverge.
pub async fn handle_messages(
    req: HttpRequest,
    body: web::Bytes,
    client: web::Data<Client>,
) -> HttpResponse {
    handle_messages_inner(req, body, client.get_ref(), None).await
}

async fn handle_messages_inner(
    req: HttpRequest,
    body: web::Bytes,
    client: &Client,
    target_override: Option<Target>,
) -> HttpResponse {
    let Some(operation) = operation_for_path(req.path()) else {
        return reject_namespace(req).await;
    };
    let raw_body = body.to_vec();
    let parsed = match serde_json::from_slice::<Value>(&raw_body) {
        Ok(value) => value,
        Err(_) => {
            return anthropic_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "request body must be valid JSON",
            );
        }
    };
    let Some(object) = parsed.as_object() else {
        return anthropic_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "request body must be a JSON object",
        );
    };
    let Some(model) = object.get("model").and_then(Value::as_str) else {
        return anthropic_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "request body requires a string model",
        );
    };
    if model.is_empty() {
        return anthropic_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "request body requires a non-empty model",
        );
    }

    let target = match target_override {
        Some(target) => target,
        None => match resolve_target(model, None, operation) {
            Ok(target) => target,
            Err(response) => return *response,
        },
    };
    let forwarded_body = if target.forwarded_model == model {
        raw_body
    } else {
        match rewrite_model_value(&body, &target.forwarded_model) {
            Ok(rewritten) => rewritten,
            Err(()) => {
                return anthropic_error(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "request body model must be a JSON string",
                );
            }
        }
    };

    let credentials = match &target.auth {
        TargetAuth::Claude => match caller_credentials(&req) {
            Some(credentials) => Some(credentials),
            None => {
                return anthropic_error(
                    StatusCode::UNAUTHORIZED,
                    "authentication_error",
                    "Claude Messages requires Authorization: Bearer or x-api-key",
                );
            }
        },
        TargetAuth::RouteKey(path) => match load_key(path) {
            Ok(key) if !key.is_empty() => Some(RouteCredential::Bearer(key)),
            Ok(_) | Err(_) => {
                log(&format!("messages missing key route={}", target.route_name));
                return anthropic_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "api_error",
                    "codex-router is missing the configured provider key",
                );
            }
        },
    };

    let query = req
        .uri()
        .query()
        .map(|value| format!("?{value}"))
        .unwrap_or_default();
    let url = format!(
        "{}{}{}",
        target.base_url.trim_end_matches('/'),
        target.path,
        query
    );
    let mut request = client.post(url);
    request = copy_request_headers(request, &req, credentials.as_ref());
    request = request.body(forwarded_body);

    let upstream = match request.send().await {
        Ok(response) => response,
        Err(error) => {
            log(&format!(
                "messages connect failure route={} model={} error={}",
                target.route_name, model, error
            ));
            return anthropic_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                "codex-router could not connect to the Messages provider",
            );
        }
    };

    let status =
        StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    log(&format!(
        "messages upstream route={} model={} status={}",
        target.route_name,
        model,
        status.as_u16()
    ));

    let mut response = HttpResponse::build(status);
    for (name, value) in upstream.headers() {
        let lower = name.as_str().to_ascii_lowercase();
        // Content-Length is intentionally omitted: Actix owns framing for the
        // stream, while protocol headers (including SSE and request ids) stay.
        if HOP_BY_HOP.contains(&lower.as_str()) || lower == "content-length" {
            continue;
        }
        if let Ok(value) = value.to_str() {
            response.append_header((name.as_str(), value));
        }
    }

    let stream = upstream.bytes_stream().map(|chunk| {
        chunk.map_err(|error| {
            actix_web::error::ErrorBadGateway(format!("Messages upstream stream failed: {error}"))
        })
    });
    response.streaming(stream)
}

#[derive(Clone)]
enum RouteCredential {
    /// Preserve a first-party Authorization header exactly as supplied.
    Authorization(String),
    /// Preserve a first-party x-api-key header exactly as supplied.
    ApiKey(String),
    /// Replace all caller credentials with the route's configured bearer key.
    Bearer(String),
}

fn resolve_target(
    model: &str,
    route_override: Option<Route>,
    operation: Operation,
) -> Result<Target, Box<HttpResponse>> {
    let lower = model.to_ascii_lowercase();
    if lower.starts_with("claude-") && model.len() > "claude-".len() {
        return Ok(Target {
            route_name: "anthropic",
            base_url: "https://api.anthropic.com".to_string(),
            path: match operation {
                Operation::Messages => "/v1/messages",
                Operation::CountTokens => "/v1/messages/count_tokens",
            },
            forwarded_model: model.to_string(),
            auth: TargetAuth::Claude,
        });
    }

    let route = route_override.or_else(|| find_route(model));
    let Some(route) = route else {
        return Err(Box::new(anthropic_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "unsupported model for Anthropic Messages",
        )));
    };
    if route.name == "muse" {
        return Err(Box::new(anthropic_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "native Anthropic Messages is not verified for Muse models",
        )));
    }
    if model.len() <= route.prefix.len() {
        return Err(Box::new(anthropic_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "unsupported model for Anthropic Messages",
        )));
    }
    let path = match operation {
        Operation::Messages => route.messages_path(),
        Operation::CountTokens => route.count_tokens_path(),
    };
    let Some(path) = path else {
        let message = if operation == Operation::CountTokens && route.name == "openrouter" {
            "Anthropic Messages count_tokens is not supported for OpenRouter models"
        } else {
            "unsupported model for Anthropic Messages"
        };
        return Err(Box::new(anthropic_error(
            if operation == Operation::CountTokens && route.name == "openrouter" {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::BAD_REQUEST
            },
            if operation == Operation::CountTokens && route.name == "openrouter" {
                "not_found_error"
            } else {
                "invalid_request_error"
            },
            message,
        )));
    };

    let forwarded_model = route
        .upstream_model(model)
        .unwrap_or_else(|| model.to_string());
    Ok(Target {
        route_name: route.name,
        base_url: route.base_url,
        path,
        forwarded_model,
        auth: TargetAuth::RouteKey(route.key_file),
    })
}

fn operation_for_path(path: &str) -> Option<Operation> {
    ROUTE_SPECS
        .iter()
        .find(|spec| spec.path == path)
        .map(|spec| spec.operation)
}

/// Rejects an unmatched path or method inside the reserved Claude namespace.
pub async fn reject_namespace(_req: HttpRequest) -> HttpResponse {
    anthropic_error(
        StatusCode::NOT_FOUND,
        "not_found_error",
        "unsupported Claude Messages endpoint",
    )
}

fn caller_credentials(req: &HttpRequest) -> Option<RouteCredential> {
    req.headers()
        .get("authorization")
        .and_then(|value| value.to_str().ok())
        .filter(|value| {
            value.split_once(' ').is_some_and(|(scheme, token)| {
                scheme.eq_ignore_ascii_case("bearer") && !token.is_empty()
            })
        })
        .map(|value| RouteCredential::Authorization(value.to_string()))
        .or_else(|| {
            req.headers()
                .get("x-api-key")
                .and_then(|value| value.to_str().ok())
                .filter(|value| !value.is_empty())
                .map(|value| RouteCredential::ApiKey(value.to_string()))
        })
}

fn copy_request_headers(
    mut request: reqwest::RequestBuilder,
    req: &HttpRequest,
    credential: Option<&RouteCredential>,
) -> reqwest::RequestBuilder {
    for (name, value) in req.headers() {
        let lower = name.as_str().to_ascii_lowercase();
        if HOP_BY_HOP.contains(&lower.as_str())
            || lower == "host"
            || lower == "content-length"
            // Actix may have already decompressed the body before this handler
            // sees it; never forward stale representation metadata upstream.
            || lower == "content-encoding"
            || lower == "authorization"
            || lower == "x-api-key"
            || lower == "cookie"
            || lower.starts_with("chatgpt-")
            || lower.starts_with("session_")
        {
            continue;
        }
        if let Ok(value) = value.to_str() {
            request = request.header(name.as_str(), value);
        }
    }
    match credential {
        Some(RouteCredential::Authorization(value)) => request.header("authorization", value),
        Some(RouteCredential::ApiKey(value)) => request.header("x-api-key", value),
        Some(RouteCredential::Bearer(value)) => request.bearer_auth(value),
        None => request,
    }
}

fn anthropic_error(status: StatusCode, error_type: &str, message: &str) -> HttpResponse {
    HttpResponse::build(status)
        .content_type("application/json")
        .json(json!({
            "type": "error",
            "error": { "type": error_type, "message": message }
        }))
}

/// Rewrites only the root JSON `model` string. Keeping the original slices for
/// every other byte preserves whitespace, content blocks, escapes, and unknown
/// fields exactly as the caller sent them.
fn rewrite_model_value(raw: &[u8], replacement: &str) -> Result<Vec<u8>, ()> {
    let mut cursor = skip_whitespace(raw, 0);
    if raw.get(cursor) != Some(&b'{') {
        return Err(());
    }
    cursor += 1;
    let mut span = None;
    loop {
        cursor = skip_whitespace(raw, cursor);
        if raw.get(cursor) == Some(&b'}') {
            break;
        }
        let key_start = cursor;
        let key_end = scan_string(raw, cursor)?;
        let key: String = serde_json::from_slice(&raw[key_start..key_end]).map_err(|_| ())?;
        cursor = skip_whitespace(raw, key_end);
        if raw.get(cursor) != Some(&b':') {
            return Err(());
        }
        cursor = skip_whitespace(raw, cursor + 1);
        let value_start = cursor;
        let value_end = scan_value(raw, cursor)?;
        if key == "model" {
            if raw.get(value_start) != Some(&b'"') {
                return Err(());
            }
            let _: String = serde_json::from_slice(&raw[value_start..value_end]).map_err(|_| ())?;
            span = Some((value_start, value_end));
        }
        cursor = skip_whitespace(raw, value_end);
        match raw.get(cursor) {
            Some(b',') => cursor += 1,
            Some(b'}') => break,
            _ => return Err(()),
        }
    }
    let (start, end) = span.ok_or(())?;
    let encoded = serde_json::to_vec(replacement).map_err(|_| ())?;
    let mut output = Vec::with_capacity(raw.len() + encoded.len().saturating_sub(end - start));
    output.extend_from_slice(&raw[..start]);
    output.extend_from_slice(&encoded);
    output.extend_from_slice(&raw[end..]);
    Ok(output)
}

fn skip_whitespace(raw: &[u8], mut cursor: usize) -> usize {
    while raw
        .get(cursor)
        .is_some_and(|byte| byte.is_ascii_whitespace())
    {
        cursor += 1;
    }
    cursor
}

fn scan_string(raw: &[u8], start: usize) -> Result<usize, ()> {
    if raw.get(start) != Some(&b'"') {
        return Err(());
    }
    let mut cursor = start + 1;
    while let Some(byte) = raw.get(cursor) {
        match byte {
            b'\\' => cursor = cursor.checked_add(2).ok_or(())?,
            b'"' => return Ok(cursor + 1),
            _ => cursor += 1,
        }
    }
    Err(())
}

fn scan_value(raw: &[u8], start: usize) -> Result<usize, ()> {
    match raw.get(start) {
        Some(b'"') => scan_string(raw, start),
        Some(b'{') | Some(b'[') => scan_compound(raw, start),
        Some(_) => {
            let mut cursor = start;
            while let Some(byte) = raw.get(cursor) {
                if matches!(byte, b',' | b'}' | b']') || byte.is_ascii_whitespace() {
                    break;
                }
                cursor += 1;
            }
            (cursor > start).then_some(cursor).ok_or(())
        }
        None => Err(()),
    }
}

fn scan_compound(raw: &[u8], start: usize) -> Result<usize, ()> {
    let opening = *raw.get(start).ok_or(())?;
    let closing = match opening {
        b'{' => b'}',
        b'[' => b']',
        _ => return Err(()),
    };
    let mut depth = 1usize;
    let mut cursor = start + 1;
    while cursor < raw.len() {
        match raw[cursor] {
            b'"' => cursor = scan_string(raw, cursor)?,
            byte if byte == opening => {
                depth += 1;
                cursor += 1;
            }
            byte if byte == closing => {
                depth -= 1;
                cursor += 1;
                if depth == 0 {
                    return Ok(cursor);
                }
            }
            _ => cursor += 1,
        }
    }
    Err(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::body::to_bytes;
    use actix_web::test::TestRequest;
    use serde_json::json;
    use std::time::{SystemTime, UNIX_EPOCH};
    use wiremock::matchers::{body_bytes, header, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn write_temp_key(value: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("codex-router-messages-key-{nanos}"));
        std::fs::write(&path, value).expect("write key");
        path
    }

    fn route(name: &'static str, base_url: String, key_file: PathBuf) -> Route {
        Route {
            name,
            prefix: match name {
                "openrouter" => "openrouter/",
                "glm" => "glm-",
                _ => "deepseek-",
            },
            base_url,
            path: "/unused",
            key_file,
            translate: crate::routes::Translate::None,
            max_body_bytes: None,
            strip_prefix: name == "openrouter",
            label_sse_events: false,
        }
    }

    fn target(model: &str, route: Route) -> Target {
        resolve_target(model, Some(route), Operation::Messages).expect("target")
    }

    #[test]
    fn model_rewrite_preserves_every_other_byte() {
        let raw =
            br#"{ "model" : "openrouter/vendor/model", "messages": [ { "content": ["a\n b"] } ] }"#;
        let rewritten = rewrite_model_value(raw, "vendor/model").expect("rewrite");
        assert_eq!(
            String::from_utf8(rewritten).expect("utf8"),
            r#"{ "model" : "vendor/model", "messages": [ { "content": ["a\n b"] } ] }"#
        );
    }

    #[test]
    fn aliases_share_the_single_handler_paths() {
        assert_eq!(CANONICAL_PATH, "/backend-api/claude");
        assert_eq!(V1_PATH, "/backend-api/claude/v1/messages");
        assert_eq!(ROOT_V1_PATH, "/v1/messages");
        assert_eq!(
            CANONICAL_COUNT_TOKENS_PATH,
            "/backend-api/claude/count_tokens"
        );
        assert_eq!(
            V1_COUNT_TOKENS_PATH,
            "/backend-api/claude/v1/messages/count_tokens"
        );
        assert_eq!(ROOT_V1_COUNT_TOKENS_PATH, "/v1/messages/count_tokens");
        assert_eq!(operation_for_path(V1_PATH), Some(Operation::Messages));
        assert_eq!(
            operation_for_path(V1_COUNT_TOKENS_PATH),
            Some(Operation::CountTokens)
        );
    }

    #[test]
    fn count_tokens_target_paths_are_provider_specific() {
        let deepseek = target(
            "deepseek-chat",
            route(
                "deepseek",
                "http://deepseek.test".to_string(),
                write_temp_key("k"),
            ),
        );
        assert_eq!(deepseek.path, "/anthropic/v1/messages");
        let deepseek = resolve_target(
            "deepseek-chat",
            Some(route(
                "deepseek",
                "http://deepseek.test".to_string(),
                write_temp_key("k"),
            )),
            Operation::CountTokens,
        )
        .expect("deepseek count target");
        assert_eq!(deepseek.path, "/anthropic/v1/messages/count_tokens");
        let glm = resolve_target(
            "glm-5.3-flash",
            Some(route(
                "glm",
                "http://glm.test".to_string(),
                write_temp_key("k"),
            )),
            Operation::CountTokens,
        )
        .expect("glm count target");
        assert_eq!(glm.path, "/api/anthropic/v1/messages/count_tokens");
        let claude = resolve_target("claude-sonnet-4-5", None, Operation::CountTokens)
            .expect("claude count target");
        assert_eq!(claude.path, "/v1/messages/count_tokens");
    }

    #[actix_web::test]
    async fn third_party_rewrites_model_for_openrouter_and_isolates_auth() {
        let server = MockServer::start().await;
        let original = br#"{ "model" : "openrouter/vendor/model", "messages": [{"role":"user","content":"hi"}] }"#;
        let forwarded =
            br#"{ "model" : "vendor/model", "messages": [{"role":"user","content":"hi"}] }"#;
        Mock::given(method("POST"))
            .and(path("/api/v1/messages"))
            .and(query_param("beta", "true"))
            .and(header("authorization", "Bearer route-key"))
            .and(header("anthropic-version", "2023-06-01"))
            .and(body_bytes(forwarded.to_vec()))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/json")
                    .insert_header("content-encoding", "gzip")
                    .set_body_raw(
                        br#"{"id":"msg_1","type":"message","content":[]}"#,
                        "application/json",
                    ),
            )
            .mount(&server)
            .await;

        let route = route("openrouter", server.uri(), write_temp_key("route-key"));
        let target = target("openrouter/vendor/model", route);
        let req = TestRequest::post()
            .uri("/backend-api/claude?beta=true")
            .insert_header(("authorization", "Bearer caller-secret"))
            .insert_header(("x-api-key", "caller-api-key"))
            .insert_header(("cookie", "session=secret"))
            .insert_header(("content-encoding", "gzip"))
            .insert_header(("anthropic-version", "2023-06-01"))
            .to_http_request();
        let response = handle_messages_inner(
            req,
            web::Bytes::from_static(original),
            &Client::new(),
            Some(target),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get("content-encoding")
                .and_then(|value| value.to_str().ok()),
            Some("gzip")
        );
        let value: Value =
            serde_json::from_slice(&to_bytes(response.into_body()).await.expect("body"))
                .expect("json");
        assert_eq!(value["id"], "msg_1");
        let received = server.received_requests().await.expect("recorded request");
        let headers = &received[0].headers;
        assert_eq!(
            headers
                .get("authorization")
                .and_then(|value| value.to_str().ok()),
            Some("Bearer route-key")
        );
        assert!(!headers.contains_key("x-api-key"));
        assert!(!headers.contains_key("cookie"));
        assert!(!headers.contains_key("content-encoding"));
    }

    #[actix_web::test]
    async fn streamed_success_and_error_bodies_are_relayed_exactly() {
        let server = MockServer::start().await;
        let sse = ": ping\n\nevent: message_start\ndata: {\"type\":\"message_start\"}\n\n";
        Mock::given(method("POST"))
            .and(path("/anthropic/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"))
            .mount(&server)
            .await;
        let deepseek_target = target(
            "deepseek-chat",
            route("deepseek", server.uri(), write_temp_key("deep-key")),
        );
        let req = TestRequest::post().uri(ROOT_V1_PATH).to_http_request();
        let body = web::Bytes::from_static(br#"{"model":"deepseek-chat","messages":[]}"#);
        let response =
            handle_messages_inner(req, body, &Client::new(), Some(deepseek_target)).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            to_bytes(response.into_body()).await.expect("body"),
            sse.as_bytes()
        );

        let error_server = MockServer::start().await;
        let error = r#"{"type":"error","error":{"type":"invalid_request_error","message":"bad"}}"#;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(400).set_body_raw(error, "application/json"))
            .mount(&error_server)
            .await;
        let glm_target = target(
            "glm-5.3-flash",
            route("glm", error_server.uri(), write_temp_key("glm-key")),
        );
        let req = TestRequest::post().uri(ROOT_V1_PATH).to_http_request();
        let body = web::Bytes::from_static(br#"{"model":"glm-5.3-flash","messages":[]}"#);
        let response = handle_messages_inner(req, body, &Client::new(), Some(glm_target)).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            to_bytes(response.into_body()).await.expect("body"),
            error.as_bytes()
        );
    }

    #[actix_web::test]
    async fn deepseek_count_tokens_forwards_query_and_openrouter_is_local_404() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/anthropic/v1/messages/count_tokens"))
            .and(query_param("beta", "true"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "input_tokens": 4
            })))
            .mount(&server)
            .await;
        let target = resolve_target(
            "deepseek-chat",
            Some(route("deepseek", server.uri(), write_temp_key("deep-key"))),
            Operation::CountTokens,
        )
        .expect("deepseek count target");
        let req = TestRequest::post()
            .uri("/v1/messages/count_tokens?beta=true")
            .to_http_request();
        let body = web::Bytes::from_static(br#"{"model":"deepseek-chat","messages":[]}"#);
        let response = handle_messages_inner(req, body, &Client::new(), Some(target)).await;
        assert_eq!(response.status(), StatusCode::OK);

        let req = TestRequest::post()
            .uri(ROOT_V1_COUNT_TOKENS_PATH)
            .to_http_request();
        let body =
            web::Bytes::from_static(br#"{"model":"openrouter/openai/gpt-5-nano","messages":[]}"#);
        let response = handle_messages_inner(req, body, &Client::new(), None).await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let value: Value =
            serde_json::from_slice(&to_bytes(response.into_body()).await.expect("body"))
                .expect("error");
        assert_eq!(value["error"]["type"], "not_found_error");
    }

    #[actix_web::test]
    async fn malformed_missing_unsupported_and_dependency_errors_are_anthropic_shaped() {
        let client = Client::new();
        let req = TestRequest::post().uri(V1_PATH).to_http_request();
        let response =
            handle_messages_inner(req, web::Bytes::from_static(b"{"), &client, None).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let value: Value =
            serde_json::from_slice(&to_bytes(response.into_body()).await.expect("body"))
                .expect("json");
        assert_eq!(value["type"], "error");
        assert_eq!(value["error"]["type"], "invalid_request_error");

        for body in [
            json!({"messages": []}),
            json!({"model": "muse-spark-1.3", "messages": []}),
            json!({"model": "unknown-model", "messages": []}),
        ] {
            let req = TestRequest::post().uri(V1_PATH).to_http_request();
            let body = web::Bytes::from(serde_json::to_vec(&body).expect("json"));
            let response = handle_messages_inner(req, body, &client, None).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let value: Value =
                serde_json::from_slice(&to_bytes(response.into_body()).await.expect("body"))
                    .expect("json");
            assert_eq!(value["type"], "error");
            assert_eq!(value["error"]["type"], "invalid_request_error");
        }

        let deepseek_target = target(
            "deepseek-chat",
            route(
                "deepseek",
                "http://127.0.0.1:1".to_string(),
                std::env::temp_dir().join("no-messages-key"),
            ),
        );
        let req = TestRequest::post().uri(V1_PATH).to_http_request();
        let body = web::Bytes::from_static(br#"{"model":"deepseek-chat","messages":[]}"#);
        let response = handle_messages_inner(req, body, &client, Some(deepseek_target)).await;
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[actix_web::test]
    async fn claude_requires_caller_key_and_uses_anthropic_headers() {
        let client = Client::new();
        let req = TestRequest::post().uri(V1_PATH).to_http_request();
        let body = web::Bytes::from_static(br#"{"model":"claude-sonnet-4-5","messages":[]}"#);
        let response = handle_messages_inner(req, body, &client, None).await;
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("authorization", "Bearer caller-secret"))
            .and(header("anthropic-beta", "prompt-caching-2024-07-31"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"type":"message"})))
            .mount(&server)
            .await;
        let req = TestRequest::post()
            .uri(ROOT_V1_PATH)
            .insert_header(("authorization", "Bearer caller-secret"))
            .insert_header(("anthropic-beta", "prompt-caching-2024-07-31"))
            .to_http_request();
        let request = copy_request_headers(
            Client::new().post(format!("{}/v1/messages", server.uri())),
            &req,
            caller_credentials(&req).as_ref(),
        )
        .body(br#"{"model":"claude-sonnet-4-5","messages":[]}"#.as_slice());
        let response = request.send().await.expect("mock response");
        assert_eq!(response.status(), reqwest::StatusCode::OK);
    }
}
