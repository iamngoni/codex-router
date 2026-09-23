use actix_web::body::to_bytes;
use actix_web::test::TestRequest;
use actix_web::{App, http::StatusCode, web};
use codex_router::models::ModelsState;
use serde_json::{Value, json};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use wiremock::matchers::{header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn temp_catalog() -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    std::env::temp_dir().join(format!("codex-router-models-{stamp}.json"))
}

fn native_body() -> &'static str {
    r#"{"models":[{"slug":"native","display_name":"Native","priority":7,"hidden":{"z":1,"a":2}},{"slug":"hidden-native","priority":42}],"unknown":{"second":2,"first":1}}"#
}

async fn response(
    server: &MockServer,
    state: web::Data<ModelsState>,
    uri: &str,
) -> (StatusCode, Value, Option<String>) {
    let request = TestRequest::get()
        .uri(uri)
        .insert_header(("authorization", "Bearer caller"))
        .insert_header(("chatgpt-account-id", "acct"))
        .insert_header(("if-none-match", "\"stale\""))
        .to_http_request();
    let client = web::Data::new(reqwest::Client::new());
    let result = codex_router::models::handle(request, client, state).await;
    let status = result.status();
    let etag = result
        .headers()
        .get("etag")
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    let body = to_bytes(result.into_body()).await.expect("body");
    let value = serde_json::from_slice(&body).unwrap_or_else(|_| {
        json!({
            "raw": String::from_utf8_lossy(&body).to_string()
        })
    });
    let _ = server;
    (status, value, etag)
}

#[actix_web::test]
async fn native_only_first_load_failure_and_merge_preserve_order() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/backend-api/codex/models"))
        .and(query_param("cursor", "one"))
        .and(header("authorization", "Bearer caller"))
        .and(header("chatgpt-account-id", "acct"))
        .and(header("accept-encoding", "identity"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("cache-control", "max-age=10")
                .insert_header("etag", "upstream")
                .insert_header("content-encoding", "gzip")
                .set_body_string(native_body()),
        )
        .expect(2)
        .mount(&server)
        .await;

    let path = temp_catalog();
    let state = web::Data::new(ModelsState::with_paths(path.clone(), server.uri()));
    let (status, first, first_etag) = response(
        &server,
        state.clone(),
        "/backend-api/codex/models?cursor=one",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(first["models"].as_array().expect("models").len(), 2);
    assert!(first_etag.is_some());

    std::fs::write(
        &path,
        json!({
            "models": [
                {"slug": "native", "display_name": "external replacement"},
                {"slug": "external-a", "display_name": "A", "priority": 999},
                {"slug": "external-b", "display_name": "B"}
            ]
        })
        .to_string(),
    )
    .expect("external");
    let (status, merged, _) =
        response(&server, state, "/backend-api/codex/models?cursor=one").await;
    assert_eq!(status, StatusCode::OK);
    let models = merged["models"].as_array().expect("models");
    assert_eq!(models[0]["display_name"], "Native");
    assert_eq!(models[2]["slug"], "external-a");
    assert_eq!(models[2]["priority"], 43);
    assert_eq!(models[3]["priority"], 44);
    assert_eq!(merged["unknown"]["second"], 2);
    assert!(merged["models"][0].get("hidden").is_some());
    let _ = std::fs::remove_file(path);
}

#[actix_web::test]
async fn malformed_keeps_lkg_and_empty_clears_with_stable_etag() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_string(native_body()))
        .mount(&server)
        .await;
    let path = temp_catalog();
    std::fs::write(
        &path,
        r#"{"models":[{"slug":"external-a","display_name":"A"}]}"#,
    )
    .expect("external");
    let state = web::Data::new(ModelsState::with_paths(path.clone(), server.uri()));

    let (_, first, first_etag) =
        response(&server, state.clone(), "/backend-api/codex/models").await;
    assert_eq!(first["models"].as_array().expect("models").len(), 3);
    let (_, second, second_etag) =
        response(&server, state.clone(), "/backend-api/codex/models").await;
    assert_eq!(first_etag, second_etag);
    assert_eq!(first, second);

    std::fs::write(&path, "malformed").expect("malformed");
    let (_, retained, retained_etag) =
        response(&server, state.clone(), "/backend-api/codex/models").await;
    assert_eq!(retained["models"].as_array().expect("models").len(), 3);
    assert_eq!(retained_etag, first_etag);

    let invalid_catalogues = [
        r#"{"models":[{"slug":"bad-nested","supported_reasoning_levels":[17]}]}"#,
        r#"{"models":[{"slug":"upgrade-missing-model","upgrade":{"migration_markdown":"Move over."}}]}"#,
        r#"{"models":[{"slug":"upgrade-missing-markdown","upgrade":{"model":"next-model"}}]}"#,
        r#"{"models":[{"slug":"upgrade-bad-model-type","upgrade":{"model":7,"migration_markdown":"Move over."}}]}"#,
        r#"{"models":[{"slug":"upgrade-bad-markdown-type","upgrade":{"model":"next-model","migration_markdown":false}}]}"#,
    ];
    for invalid_catalogue in invalid_catalogues {
        std::fs::write(&path, invalid_catalogue).expect("invalid nested catalogue");
        let (_, retained, retained_etag) =
            response(&server, state.clone(), "/backend-api/codex/models").await;
        assert_eq!(retained, first);
        assert_eq!(retained_etag, first_etag);
    }

    std::fs::write(&path, r#"{"models":[]}"#).expect("empty");
    let (_, cleared, cleared_etag) = response(&server, state, "/backend-api/codex/models").await;
    assert_eq!(cleared["models"].as_array().expect("models").len(), 2);
    assert_ne!(cleared_etag, first_etag);
    let _ = std::fs::remove_file(path);
}

#[actix_web::test]
async fn nullable_fields_and_object_upgrade_are_merged() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_string(native_body()))
        .mount(&server)
        .await;
    let path = temp_catalog();
    std::fs::write(
        &path,
        json!({
            "models": [{
                "slug": "nullable-fields",
                "description": null,
                "default_reasoning_level": null,
                "context_window": null,
                "max_context_window": null,
                "auto_compact_token_limit": null,
                "model_messages": null,
                "availability_nux": null,
                "upgrade": {
                    "model": "next-model",
                    "migration_markdown": "Move over.",
                    "future_field": "preserved"
                },
                "supported_reasoning_levels": [{"effort": "high", "description": "High"}],
                "input_modalities": ["text", "image"],
                "service_tiers": [{
                    "id": "priority",
                    "name": "Fast",
                    "description": "2x speed"
                }],
                "additional_speed_tiers": ["fast"],
                "experimental_supported_tools": ["browser"]
            }]
        })
        .to_string(),
    )
    .expect("nullable external catalogue");
    let state = web::Data::new(ModelsState::with_paths(path.clone(), server.uri()));

    let (status, merged, _) = response(&server, state, "/backend-api/codex/models").await;
    assert_eq!(status, StatusCode::OK);
    let model = &merged["models"][2];
    assert_eq!(model["description"], Value::Null);
    assert_eq!(model["context_window"], Value::Null);
    assert_eq!(model["upgrade"]["model"], "next-model");
    assert_eq!(model["upgrade"]["migration_markdown"], "Move over.");
    assert_eq!(model["upgrade"]["future_field"], "preserved");
    assert_eq!(model["supported_reasoning_levels"][0]["effort"], "high");
    assert_eq!(model["input_modalities"][1], "image");
    assert_eq!(model["service_tiers"][0]["id"], "priority");
    assert_eq!(model["additional_speed_tiers"][0], "fast");
    assert_eq!(model["experimental_supported_tools"][0], "browser");
    let _ = std::fs::remove_file(path);
}

#[actix_web::test]
async fn matching_weak_conditional_etag_returns_304_and_errors_are_unmerged() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(200).set_body_string(native_body()))
        .mount(&server)
        .await;
    let path = temp_catalog();
    let state = web::Data::new(ModelsState::with_paths(path.clone(), server.uri()));
    let request = TestRequest::get()
        .uri("/backend-api/codex/models")
        .to_http_request();
    let client = web::Data::new(reqwest::Client::new());
    let first = codex_router::models::handle(request, client.clone(), state.clone()).await;
    let etag = first
        .headers()
        .get("etag")
        .expect("etag")
        .to_str()
        .expect("etag")
        .to_string();
    let conditional = TestRequest::get()
        .uri("/backend-api/codex/models")
        .insert_header(("if-none-match", format!("W/{etag}, \"other\"")))
        .to_http_request();
    let second = codex_router::models::handle(conditional, client.clone(), state.clone()).await;
    assert_eq!(second.status(), StatusCode::NOT_MODIFIED);
    assert!(to_bytes(second.into_body()).await.expect("body").is_empty());

    server.reset().await;
    Mock::given(method("GET"))
        .respond_with(ResponseTemplate::new(503).set_body_string("upstream failure"))
        .mount(&server)
        .await;
    let error_request = TestRequest::get()
        .uri("/backend-api/codex/models")
        .to_http_request();
    let error = codex_router::models::handle(error_request, client, state).await;
    assert_eq!(error.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        to_bytes(error.into_body()).await.expect("body"),
        "upstream failure"
    );
    let _ = std::fs::remove_file(path);
}

#[actix_web::test]
async fn unsolicited_upstream_304_becomes_safe_bad_gateway() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/backend-api/codex/models"))
        .respond_with(
            ResponseTemplate::new(304)
                .insert_header("etag", "upstream-validator")
                .insert_header("cache-control", "public, max-age=60"),
        )
        .expect(1)
        .mount(&server)
        .await;
    let state = web::Data::new(ModelsState::with_paths(temp_catalog(), server.uri()));
    let request = TestRequest::get()
        .uri("/backend-api/codex/models")
        .insert_header(("if-none-match", "\"caller-validator\""))
        .to_http_request();
    let response =
        codex_router::models::handle(request, web::Data::new(reqwest::Client::new()), state).await;

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert!(response.headers().get("etag").is_none());
    assert!(response.headers().get("cache-control").is_none());
    assert_eq!(
        to_bytes(response.into_body()).await.expect("body"),
        "codex-router: models upstream error"
    );
}

#[actix_web::test]
async fn dispatch_models_route_preserves_query_and_only_safe_headers() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/backend-api/codex/models"))
        .and(query_param("cursor", "one"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("cache-control", "max-age=10")
                .insert_header("etag", "upstream-validator")
                .insert_header("content-encoding", "gzip")
                .set_body_string(native_body()),
        )
        .expect(1)
        .mount(&server)
        .await;
    let state = web::Data::new(ModelsState::with_paths(temp_catalog(), server.uri()));
    let client = web::Data::new(reqwest::Client::new());
    let app = actix_web::test::init_service(
        App::new()
            .app_data(state)
            .app_data(client)
            .default_service(web::route().to(codex_router::dispatch)),
    )
    .await;
    let request = TestRequest::get()
        .uri("/backend-api/codex/models?cursor=one")
        .to_request();
    let response = actix_web::test::call_service(&app, request).await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "application/json"
    );
    assert_eq!(
        response.headers().get("cache-control").unwrap(),
        "max-age=10"
    );
    assert!(response.headers().get("content-encoding").is_none());
    assert_ne!(
        response.headers().get("etag").unwrap(),
        "upstream-validator"
    );
    let body = to_bytes(response.into_body()).await.expect("body");
    let value: Value = serde_json::from_slice(&body).expect("JSON model catalogue");
    assert_eq!(value["models"].as_array().expect("models").len(), 2);
}
