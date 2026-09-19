//! The model-name-prefix routing table.
//!
//! Owns only route *selection* — no request/response shape knowledge lives
//! here. `translate.rs` and `proxy.rs` decide what to do with the `Route`
//! this module hands back.

use crate::config::home_dir;
use std::path::PathBuf;

/// The largest request body DeepSeek accepts. Their edge (openresty in front
/// of `api.deepseek.com`) answers anything larger with a bare HTML `413`
/// before the request reaches the model, which Codex then retries a few
/// times — tens of megabytes each attempt — before giving up.
///
/// Measured against the live edge rather than taken from docs: a body of
/// exactly 50,331,648 bytes (48 MiB) is accepted (it reaches model
/// validation), one byte more is rejected. `dispatch` refuses larger bodies
/// locally so the failure costs one line instead of six uploads.
pub const DEEPSEEK_MAX_BODY_BYTES: usize = 48 * 1024 * 1024;

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
    /// Largest request body the upstream's edge accepts, where we've actually
    /// measured one — see [`DEEPSEEK_MAX_BODY_BYTES`]. `None` means "not
    /// measured", not "unlimited": the body is forwarded unchecked and the
    /// upstream decides.
    pub max_body_bytes: Option<usize>,
    /// Whether `prefix` is *Codex's* naming that the provider does not know, so
    /// it has to come off before forwarding (`openrouter/anthropic/claude-x`
    /// must reach OpenRouter as `anthropic/claude-x`). Every other route's
    /// prefix is part of the provider's own model name — `glm-5.3` is sent to
    /// Z.ai as `glm-5.3`.
    pub strip_prefix: bool,
    /// Whether this provider streams Responses events as bare `data:` frames
    /// with no `event: <type>` label in front. OpenAI and DeepSeek send the
    /// label; OpenRouter does not, and Codex was written against the label, so
    /// the passthrough adds it back (the type is already in the payload — see
    /// `proxy::label_sse_events`).
    pub label_sse_events: bool,
}

impl Route {
    /// The model name to forward, when it differs from the one Codex asked for.
    pub fn upstream_model(&self, model: &str) -> Option<String> {
        if !self.strip_prefix {
            return None;
        }
        let rest = model.get(self.prefix.len()..)?;
        (!rest.is_empty()).then(|| rest.to_string())
    }

    /// Returns the native Anthropic Messages path for providers whose route
    /// has been verified to speak that protocol. This is deliberately a
    /// method instead of another struct field so existing route fixtures keep
    /// compiling without a second, easy-to-forget configuration switch.
    pub fn messages_path(&self) -> Option<&'static str> {
        match self.name {
            "deepseek" => Some("/anthropic/v1/messages"),
            "openrouter" => Some("/api/v1/messages"),
            "glm" => Some("/api/anthropic/v1/messages"),
            "muse" => None,
            _ => None,
        }
    }

    /// Returns the native token-counting path where live compatibility has
    /// been verified. OpenRouter intentionally returns `None`: its endpoint
    /// rejects this operation, so the Messages handler returns a local 404.
    pub fn count_tokens_path(&self) -> Option<&'static str> {
        match self.name {
            "deepseek" => Some("/anthropic/v1/messages/count_tokens"),
            "glm" => Some("/api/anthropic/v1/messages/count_tokens"),
            "openrouter" | "muse" => None,
            _ => None,
        }
    }
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
            max_body_bytes: Some(DEEPSEEK_MAX_BODY_BYTES),
            strip_prefix: false,
            label_sse_events: false,
        },
        Route {
            name: "muse",
            prefix: "muse-",
            base_url: "https://api.meta.ai".to_string(),
            path: "/v1/responses",
            key_file: home.join(".config").join("meta").join("key"),
            translate: Translate::None,
            max_body_bytes: None,
            strip_prefix: false,
            label_sse_events: false,
        },
        Route {
            // OpenRouter fronts many vendors behind one OpenAI-shaped API,
            // including a Responses endpoint that streams the same event types
            // Codex expects — so this is a pass-through like DeepSeek, not a
            // translation like GLM. Model ids there are namespaced by vendor
            // (`anthropic/claude-sonnet-4.5`), so Codex asks for
            // `openrouter/anthropic/claude-sonnet-4.5` and the route's prefix is
            // stripped before forwarding.
            //
            // No measured body-size limit yet, so no local trim or refusal:
            // oversized requests reach OpenRouter and it decides.
            name: "openrouter",
            prefix: "openrouter/",
            base_url: "https://openrouter.ai".to_string(),
            path: "/api/v1/responses",
            key_file: home.join(".config").join("openrouter").join("key"),
            translate: Translate::None,
            max_body_bytes: None,
            strip_prefix: true,
            // Measured against the live API: OpenRouter's stream carries the
            // right `type`s inside `data:` frames, but no `event:` lines.
            label_sse_events: true,
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
            max_body_bytes: None,
            strip_prefix: false,
            label_sse_events: false,
        },
    ]
}

/// Returns the first route whose prefix matches `model` (case-insensitive),
/// or `None` for OpenAI's own models (which fall through to the default
/// pass-through).
pub fn find_route(model: &str) -> Option<Route> {
    let lower = model.to_lowercase();
    routes().into_iter().find(|r| lower.starts_with(r.prefix))
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

    #[test]
    fn openrouter_route_strips_its_prefix_for_the_provider() {
        let route = find_route("openrouter/anthropic/claude-sonnet-4.5").expect("routed");
        assert_eq!(route.name, "openrouter");
        assert_eq!(
            route.upstream_model("openrouter/anthropic/claude-sonnet-4.5"),
            Some("anthropic/claude-sonnet-4.5".to_string())
        );
        // A bare prefix carries no model to forward.
        assert_eq!(route.upstream_model("openrouter/"), None);
    }

    #[test]
    fn other_routes_forward_the_model_name_unchanged() {
        for model in ["deepseek-flash", "glm-5.3-flash", "muse-spark-1.3"] {
            let route = find_route(model).expect("routed");
            assert_eq!(route.upstream_model(model), None, "{model}");
        }
    }

    #[test]
    fn native_messages_paths_are_explicit_and_muse_is_not_claimed_supported() {
        assert_eq!(
            find_route("deepseek-chat").unwrap().messages_path(),
            Some("/anthropic/v1/messages")
        );
        assert_eq!(
            find_route("openrouter/anthropic/claude-sonnet")
                .unwrap()
                .messages_path(),
            Some("/api/v1/messages")
        );
        assert_eq!(
            find_route("glm-5.3-flash").unwrap().messages_path(),
            Some("/api/anthropic/v1/messages")
        );
        assert_eq!(find_route("muse-spark-1.3").unwrap().messages_path(), None);
        assert_eq!(
            find_route("deepseek-chat").unwrap().count_tokens_path(),
            Some("/anthropic/v1/messages/count_tokens")
        );
        assert_eq!(
            find_route("glm-5.3-flash").unwrap().count_tokens_path(),
            Some("/api/anthropic/v1/messages/count_tokens")
        );
        assert_eq!(
            find_route("openrouter/openai/gpt-5-nano")
                .unwrap()
                .count_tokens_path(),
            None
        );
    }

    #[test]
    fn matches_case_insensitive() {
        assert_eq!(find_route("GLM-5.3-Flash").unwrap().name, "glm");
        assert_eq!(find_route("DeepSeek-Flash").unwrap().name, "deepseek");
        assert_eq!(find_route("MUSE-Spark-1.3").unwrap().name, "muse");
    }
}
