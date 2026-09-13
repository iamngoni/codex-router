//! The model-name-prefix routing table.
//!
//! Owns only route *selection* — no request/response shape knowledge lives
//! here. `translate.rs` and `proxy.rs` decide what to do with the `Route`
//! this module hands back.

use crate::config::home_dir;
use std::path::PathBuf;

/// Whether a route's body needs shape translation before it can reach the
/// upstream, or can be forwarded as-is.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Translate {
    /// Upstream already speaks the Responses API — forward the body untouched.
    None,
    /// Upstream only speaks Chat Completions — translate in both directions.
    Chat,
}

/// One upstream provider reachable through a model-name prefix.
///
/// `base_url` carries the scheme (`https://api.z.ai`, never just a
/// hostname) so `reqwest` derives the `Host` header itself from the request
/// URL, and so a test can override it to a local mock server without any
/// other code caring.
#[derive(Clone)]
pub struct Route {
    pub name: &'static str,
    pub prefix: &'static str,
    pub base_url: String,
    pub path: &'static str,
    pub key_file: PathBuf,
    pub translate: Translate,
}

/// Each route matches on a `model` name prefix. Codex's active provider is
/// always "router" (base_url http://127.0.0.1:PORT) for any of these models;
/// this table decides which upstream a given request actually goes to.
pub fn routes() -> Vec<Route> {
    let home = home_dir();
    vec![
        Route {
            name: "deepseek",
            prefix: "deepseek-",
            base_url: "https://api.deepseek.com".to_string(),
            path: "/responses",
            key_file: home.join(".config").join("deepseek").join("key"),
            translate: Translate::None,
        },
        Route {
            name: "muse",
            prefix: "muse-",
            base_url: "https://api.meta.ai".to_string(),
            path: "/v1/responses",
            key_file: home.join(".config").join("meta").join("key"),
            translate: Translate::None,
        },
        Route {
            // Z.ai has no Responses-API endpoint — only Chat Completions
            // (`messages`, not `input`). Codex only ever builds
            // Responses-shaped requests, so this route translates in both
            // directions (see translate.rs) instead of passing the body
            // through untouched.
            name: "glm",
            prefix: "glm-",
            base_url: "https://api.z.ai".to_string(),
            path: "/api/paas/v4/chat/completions",
            key_file: home.join(".config").join("zai").join("key"),
            translate: Translate::Chat,
        },
    ]
}

/// Returns the first route whose prefix matches `model`, or `None` for
/// OpenAI's own models (which fall through to the default pass-through).
pub fn find_route(model: &str) -> Option<Route> {
    routes().into_iter().find(|r| model.starts_with(r.prefix))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_by_prefix() {
        assert_eq!(find_route("glm-5.3-flash").unwrap().name, "glm");
        assert_eq!(find_route("deepseek-flash").unwrap().name, "deepseek");
        assert_eq!(find_route("muse-spark-1.3").unwrap().name, "muse");
        assert!(find_route("gpt-6-astra").is_none());
    }
}
