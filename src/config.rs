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
