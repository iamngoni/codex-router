//! Process-wide constants and well-known on-disk paths.
//!
//! This module owns no state — every function here is a pure derivation
//! from `$HOME`, resolved fresh on each call so tests can override `HOME`
//! without any global mutable state.

use std::env;
use std::path::PathBuf;

pub const HOST: &str = "127.0.0.1";
pub const PORT: u16 = 4141;
pub const OPENAI_HOST: &str = "chatgpt.com";

/// Actix's `web::Bytes` extractor defaults to a 256 KiB body limit — fine
/// for a public-facing API, but Codex's own requests (full conversation
/// history, tool schemas, file contents) routinely exceed that long before
/// they reach any particular provider. This has to be one generous limit
/// for the whole process, not a per-route setting: the body size is capped
/// before `dispatch` ever parses `model` to pick a route.
pub const MAX_PAYLOAD_BYTES: usize = 100 * 1024 * 1024;

/// Resolves `$HOME`, falling back to `/` only if the environment is somehow
/// missing it entirely (never expected outside a stripped-down sandbox).
pub fn home_dir() -> PathBuf {
    env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/"))
}

/// Append-only request log, shared with the on-disk log an operator tails.
pub fn log_file() -> PathBuf {
    home_dir()
        .join(".local")
        .join("state")
        .join("codex-router.log")
}

/// Overwritten with the most recent upstream error body, for quick
/// inspection without grepping the full log.
pub fn last_error_file() -> PathBuf {
    home_dir()
        .join(".local")
        .join("state")
        .join("codex-router-last-error.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_paths_live_under_home() {
        assert!(log_file().starts_with(home_dir()));
        assert!(log_file().ends_with("codex-router.log"));
        assert!(last_error_file().ends_with("codex-router-last-error.json"));
    }
}
