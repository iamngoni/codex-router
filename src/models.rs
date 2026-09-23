//! Live model-catalog endpoint and its last-known-good external catalogue.

use crate::config::{OPENAI_HOST, external_catalog_file};
use actix_web::{HttpRequest, HttpResponse, http::header, web};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::io::{ErrorKind, Read};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const MAX_EXTERNAL_CATALOG_BYTES: usize = 8 * 1024 * 1024;
const MAX_NATIVE_RESPONSE_BYTES: usize = 64 * 1024 * 1024;

const HOP_BY_HOP: [&str; 7] = [
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
];

const CONDITIONAL_REQUEST_HEADERS: [&str; 5] = [
    "if-match",
    "if-none-match",
    "if-modified-since",
    "if-unmodified-since",
    "if-range",
];

/// Shared process state. The external entries are replaced only after a
/// complete, valid file has been loaded; an unreadable or invalid reload
/// leaves the previous value in place.
pub struct ModelsState {
    catalog_path: PathBuf,
    upstream_base_url: String,
    external: Arc<Mutex<Option<Vec<Value>>>>,
}

impl ModelsState {
    pub fn new() -> Self {
        Self::with_paths(external_catalog_file(), format!("https://{OPENAI_HOST}"))
    }

    pub fn with_paths(catalog_path: PathBuf, upstream_base_url: String) -> Self {
        Self {
            catalog_path,
            upstream_base_url,
            external: Arc::new(Mutex::new(None)),
        }
    }
}

impl Default for ModelsState {
    fn default() -> Self {
        Self::new()
    }
}

/// Handles the exact native model-list request. The upstream is fetched on
/// every call, even when the local external catalogue cannot be loaded.
pub async fn handle(
    req: HttpRequest,
    client: web::Data<reqwest::Client>,
    state: web::Data<ModelsState>,
) -> HttpResponse {
    let path_and_query = req
        .uri()
        .path_and_query()
        .map(|value| value.as_str())
        .unwrap_or(req.path());
    let url = format!("{}{}", state.upstream_base_url, path_and_query);
    let mut builder = client.get(url);

    for (name, value) in req.headers() {
        let lower = name.as_str().to_ascii_lowercase();
        if lower == "host"
            || lower == "content-length"
            || lower == "content-encoding"
            || lower == "accept-encoding"
            || HOP_BY_HOP.contains(&lower.as_str())
            || CONDITIONAL_REQUEST_HEADERS.contains(&lower.as_str())
        {
            continue;
        }
        if let Ok(value) = value.to_str() {
            builder = builder.header(name.as_str(), value);
        }
    }
    builder = builder.header("accept-encoding", "identity");

    let upstream = match builder.send().await {
        Ok(response) => response,
        Err(_) => {
            crate::logging::log("models upstream error");
            return bad_gateway();
        }
    };
    let status = upstream.status();
    let headers = upstream.headers().clone();
    if status == reqwest::StatusCode::NOT_MODIFIED {
        crate::logging::log("models upstream returned unsolicited 304");
        return bad_gateway();
    }
    let body = match read_bounded_response(upstream).await {
        Ok(body) => body,
        Err(_) => {
            crate::logging::log("models upstream body error");
            return bad_gateway();
        }
    };

    if !status.is_success() {
        return response_with_safe_headers(status.as_u16(), &headers, body, None);
    }

    let native = match parse_native(&body) {
        Ok(native) => native,
        Err(_) => return bad_gateway(),
    };

    match state.reload_external().await {
        Ok(count) => crate::logging::log(&format!(
            "models external catalogue reload succeeded entries={count}"
        )),
        Err(reason) => crate::logging::log(&format!(
            "models external catalogue reload failed reason={reason}; retaining last-known-good"
        )),
    }
    let merged = state.merge_native(native);
    let etag = quoted_sha256(&merged);
    if req
        .headers()
        .get(header::IF_NONE_MATCH)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| if_none_match(value, &etag))
    {
        return response_with_safe_headers(304, &headers, Vec::new(), Some(&etag));
    }

    response_with_safe_headers(200, &headers, merged, Some(&etag))
}

impl ModelsState {
    async fn reload_external(&self) -> Result<usize, &'static str> {
        let path = self.catalog_path.clone();
        let external = Arc::clone(&self.external);
        actix_web::rt::task::spawn_blocking(move || {
            let mut guard = external
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let bytes = read_bounded_file(&path).map_err(|_| "file")?;
            let models = parse_external(&bytes).ok_or("validation")?;
            let count = models.len();
            *guard = Some(models);
            Ok(count)
        })
        .await
        .map_err(|_| "worker")?
    }

    fn merge_native(&self, mut native: Value) -> Vec<u8> {
        let external = self
            .external
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();

        if let Some(external) = external {
            let mut candidate = native.clone();
            if let Ok(()) = append_external(&mut candidate, external) {
                // The native object and every native record remain untouched;
                // only the added records' priorities are assigned here.
                native = candidate;
            }
        }
        serde_json::to_vec(&native).unwrap_or_else(|_| b"{}".to_vec())
    }
}

async fn read_bounded_response(response: reqwest::Response) -> Result<Vec<u8>, ()> {
    read_bounded_response_with_limit(response, MAX_NATIVE_RESPONSE_BYTES).await
}

async fn read_bounded_response_with_limit(
    mut response: reqwest::Response,
    max_bytes: usize,
) -> Result<Vec<u8>, ()> {
    if response
        .content_length()
        .is_some_and(|length| length > max_bytes as u64)
    {
        return Err(());
    }
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|_| ())? {
        if body.len().saturating_add(chunk.len()) > max_bytes {
            return Err(());
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn read_bounded_file(path: &Path) -> std::io::Result<Vec<u8>> {
    let file = std::fs::File::open(path)?;
    let mut body = Vec::new();
    file.take((MAX_EXTERNAL_CATALOG_BYTES + 1) as u64)
        .read_to_end(&mut body)?;
    if body.len() > MAX_EXTERNAL_CATALOG_BYTES {
        return Err(std::io::Error::new(
            ErrorKind::InvalidData,
            "external catalogue exceeds size limit",
        ));
    }
    Ok(body)
}

fn parse_native(body: &[u8]) -> Result<Value, ()> {
    let value: Value = serde_json::from_slice(body).map_err(|_| ())?;
    let models = value
        .as_object()
        .and_then(|object| object.get("models"))
        .and_then(Value::as_array)
        .ok_or(())?;
    if models.iter().all(Value::is_object) {
        Ok(value)
    } else {
        Err(())
    }
}

fn parse_external(body: &[u8]) -> Option<Vec<Value>> {
    let value: Value = serde_json::from_slice(body).ok()?;
    let models = value.as_object()?.get("models")?.as_array()?;
    let mut slugs = HashSet::with_capacity(models.len());
    for model in models {
        let object = model.as_object()?;
        let slug = object.get("slug")?.as_str()?.trim();
        if slug.is_empty() || !slugs.insert(slug.to_string()) || !sensible_fields(object) {
            return None;
        }
    }
    Some(models.clone())
}

fn sensible_fields(object: &Map<String, Value>) -> bool {
    const STRINGS: [&str; 12] = [
        "slug",
        "display_name",
        "base_instructions",
        "shell_type",
        "visibility",
        "default_reasoning_summary",
        "comp_hash",
        "model_specialty",
        "multi_agent_reasoning_effort",
        "multi_agent_version",
        "tool_mode",
        "web_search_tool_type",
    ];
    const OPTIONAL_STRINGS: [&str; 4] = [
        "description",
        "default_reasoning_level",
        "default_verbosity",
        "apply_patch_tool_type",
    ];
    const INTEGERS: [&str; 3] = [
        "context_window",
        "max_context_window",
        "auto_compact_token_limit",
    ];
    const REQUIRED_INTEGERS: [&str; 2] = ["priority", "effective_context_window_percent"];
    const BOOLS: [&str; 14] = [
        "supported_in_api",
        "support_verbosity",
        "supports_parallel_tool_calls",
        "use_responses_lite",
        "prefer_websockets",
        "supports_search_tool",
        "supports_image_detail_original",
        "supports_experimental_context",
        "supports_reasoning_summary_parameter",
        "include_apps_usage_instructions",
        "include_plugin_usage_instructions",
        "include_skills_usage_instructions",
        "node_repl_disabled",
        "node_repl_auto_review_required",
    ];
    const STRING_ARRAYS: [&str; 3] = [
        "additional_speed_tiers",
        "experimental_supported_tools",
        "input_modalities",
    ];
    STRINGS
        .iter()
        .all(|key| field_is(object, key, Value::is_string))
        && OPTIONAL_STRINGS
            .iter()
            .all(|key| field_is_or_null(object, key, Value::is_string))
        && INTEGERS
            .iter()
            .all(|key| field_is_or_null(object, key, |value| value.as_i64().is_some()))
        && REQUIRED_INTEGERS.iter().all(|key| {
            object
                .get(*key)
                .is_none_or(|value| value.as_i64().is_some())
        })
        && object.get("priority").is_none_or(|value| {
            value
                .as_i64()
                .is_some_and(|number| i32::try_from(number).is_ok())
        })
        && BOOLS
            .iter()
            .all(|key| field_is(object, key, Value::is_boolean))
        && field_is_array_of(
            object,
            "supported_reasoning_levels",
            reasoning_level_is_valid,
        )
        && field_is_array_of(object, "service_tiers", service_tier_is_valid)
        && STRING_ARRAYS
            .iter()
            .all(|key| field_is_array_of(object, key, Value::is_string))
        && field_is(object, "truncation_policy", Value::is_object)
        && field_is_or_null(object, "model_messages", Value::is_object)
        && field_is_or_null(object, "availability_nux", availability_nux_is_valid)
        && field_is_or_null(object, "upgrade", upgrade_is_valid)
        && field_is(object, "truncation_policy", truncation_policy_is_valid)
        && field_is_or_null(object, "model_messages", model_messages_is_valid)
}

fn field_is(object: &Map<String, Value>, key: &str, predicate: fn(&Value) -> bool) -> bool {
    object.get(key).is_none_or(predicate)
}

fn field_is_or_null(object: &Map<String, Value>, key: &str, predicate: fn(&Value) -> bool) -> bool {
    object
        .get(key)
        .is_none_or(|value| value.is_null() || predicate(value))
}

fn field_is_array_of(
    object: &Map<String, Value>,
    key: &str,
    predicate: fn(&Value) -> bool,
) -> bool {
    object.get(key).is_none_or(|value| {
        value
            .as_array()
            .is_some_and(|items| items.iter().all(predicate))
    })
}

fn reasoning_level_is_valid(value: &Value) -> bool {
    value.as_object().is_some_and(|object| {
        object.get("effort").is_some_and(Value::is_string)
            && object.get("description").is_some_and(Value::is_string)
    })
}

fn service_tier_is_valid(value: &Value) -> bool {
    value.as_object().is_some_and(|object| {
        ["id", "name", "description"]
            .iter()
            .all(|key| object.get(*key).is_some_and(Value::is_string))
    })
}

fn availability_nux_is_valid(value: &Value) -> bool {
    value
        .as_object()
        .is_some_and(|object| object.get("message").is_some_and(Value::is_string))
}

fn upgrade_is_valid(value: &Value) -> bool {
    value.as_object().is_some_and(|object| {
        object.get("model").is_some_and(Value::is_string)
            && object
                .get("migration_markdown")
                .is_some_and(Value::is_string)
    })
}

fn truncation_policy_is_valid(value: &Value) -> bool {
    value.as_object().is_some_and(|object| {
        object.get("mode").is_some_and(Value::is_string)
            && object
                .get("limit")
                .is_some_and(|limit| limit.as_i64().is_some())
    })
}

fn model_messages_is_valid(value: &Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    let template_valid = object
        .get("instructions_template")
        .is_none_or(|template| template.is_null() || template.is_string());
    let variables_valid = object
        .get("instructions_variables")
        .is_none_or(|variables| {
            variables.is_null()
                || variables.as_object().is_some_and(|variables| {
                    [
                        "personality_default",
                        "personality_friendly",
                        "personality_pragmatic",
                    ]
                    .iter()
                    .all(|key| {
                        variables
                            .get(*key)
                            .is_none_or(|value| value.is_null() || value.is_string())
                    })
                })
        });
    template_valid && variables_valid
}

fn append_external(native: &mut Value, external: Vec<Value>) -> Result<(), ()> {
    let object = native.as_object_mut().ok_or(())?;
    let native_models = object
        .get_mut("models")
        .and_then(Value::as_array_mut)
        .ok_or(())?;
    let mut native_slugs = HashSet::with_capacity(native_models.len());
    let mut max_priority = None;
    for model in native_models.iter() {
        let record = model.as_object().ok_or(())?;
        if let Some(slug) = record.get("slug").and_then(Value::as_str) {
            native_slugs.insert(slug.to_string());
        }
        if let Some(priority) = record.get("priority") {
            let value = priority
                .as_i64()
                .and_then(|number| i32::try_from(number).ok())
                .ok_or(())?;
            max_priority = Some(max_priority.map_or(value, |current: i32| current.max(value)));
        }
    }

    let mut next_priority = max_priority.unwrap_or(0);
    for mut model in external {
        let slug = model.get("slug").and_then(Value::as_str).ok_or(())?;
        if native_slugs.contains(slug) {
            continue;
        }
        next_priority = next_priority.checked_add(1).ok_or(())?;
        model
            .as_object_mut()
            .ok_or(())?
            .insert("priority".to_string(), Value::from(next_priority));
        native_models.push(model);
    }
    Ok(())
}

fn quoted_sha256(body: &[u8]) -> String {
    let digest = Sha256::digest(body);
    format!("\"{digest:x}\"")
}

fn if_none_match(header_value: &str, current: &str) -> bool {
    header_value.split(',').any(|candidate| {
        let candidate = candidate.trim();
        candidate == "*" || candidate.strip_prefix("W/").unwrap_or(candidate) == current
    })
}

fn response_with_safe_headers(
    status: u16,
    upstream_headers: &reqwest::header::HeaderMap,
    body: Vec<u8>,
    etag: Option<&str>,
) -> HttpResponse {
    let code = actix_web::http::StatusCode::from_u16(status)
        .unwrap_or(actix_web::http::StatusCode::BAD_GATEWAY);
    let mut response = HttpResponse::build(code);
    for (name, value) in upstream_headers {
        let lower = name.as_str().to_ascii_lowercase();
        if HOP_BY_HOP.contains(&lower.as_str())
            || matches!(
                lower.as_str(),
                "content-length"
                    | "content-encoding"
                    | "etag"
                    | "last-modified"
                    | "digest"
                    | "content-md5"
            )
        {
            continue;
        }
        if lower == "content-type" {
            if status >= 400
                && let Ok(value) = value.to_str()
            {
                response.append_header((name.as_str(), value));
            }
            continue;
        }
        if matches!(
            lower.as_str(),
            "cache-control" | "expires" | "pragma" | "vary" | "age" | "date"
        ) && let Ok(value) = value.to_str()
        {
            response.append_header((name.as_str(), value));
        }
    }
    if status == 200 {
        response.insert_header((header::CONTENT_TYPE, "application/json"));
    }
    if let Some(etag) = etag {
        response.insert_header((header::ETAG, etag));
    }
    response.body(body)
}

fn bad_gateway() -> HttpResponse {
    HttpResponse::BadGateway()
        .content_type("text/plain")
        .body("codex-router: models upstream error")
}

#[cfg(test)]
mod tests {
    use super::read_bounded_response_with_limit;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    use std::thread;
    use std::time::Duration;

    #[actix_web::test]
    async fn streaming_body_limit_fails_before_chunked_response_ends() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind listener");
        let address = listener.local_addr().expect("listener address");
        let (release_tx, release_rx) = mpsc::channel();
        let upstream_open = Arc::new(AtomicBool::new(true));
        let server_upstream_open = Arc::clone(&upstream_open);
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept client");
            let mut request = Vec::new();
            let mut buffer = [0_u8; 512];
            while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                let read = stream.read(&mut buffer).expect("read request");
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
            }
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n8\r\n12345678\r\n",
                )
                .expect("write oversized chunk");
            stream.flush().expect("flush oversized chunk");
            let _ = release_rx.recv_timeout(Duration::from_secs(2));
            server_upstream_open.store(false, Ordering::SeqCst);
        });

        let response = reqwest::Client::new()
            .get(format!("http://{address}/models"))
            .send()
            .await
            .expect("upstream response headers");
        assert!(read_bounded_response_with_limit(response, 4).await.is_err());
        let upstream_remained_open = upstream_open.load(Ordering::SeqCst);
        let _ = release_tx.send(());
        server.join().expect("server thread");
        assert!(upstream_remained_open, "body limit waits for upstream EOF");
    }
}
