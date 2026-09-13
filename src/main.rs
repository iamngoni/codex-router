//! Binary entry point: builds the shared `reqwest::Client` and starts the
//! Actix server on `config::HOST:PORT`. All request-handling logic lives in
//! the library crate so integration tests exercise the identical wiring.

use actix_web::{App, HttpServer, web};
use codex_router::{config, dispatch, healthz, logging::log};

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let client = reqwest::Client::new();
    log(&format!("listening on {}:{}", config::HOST, config::PORT));

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(client.clone()))
            .app_data(web::PayloadConfig::new(config::MAX_PAYLOAD_BYTES))
            .route("/healthz", web::get().to(healthz))
            .default_service(web::route().to(dispatch))
    })
    .bind((config::HOST, config::PORT))?
    .run()
    .await
}
