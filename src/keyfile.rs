//! Reads a provider API key from a single-line file on disk.
//!
//! Keys never live in this repository or in `config.toml` — each provider's
//! key file lives under `~/.config/<provider>/key`, mode 600, outside any
//! git-tracked tree.

use std::path::Path;

/// Reads and trims a key file's contents. Returns an error rather than
/// panicking so callers can turn a missing/unreadable key into a proper
/// HTTP 500 instead of taking the process down.
pub fn load_key(path: &Path) -> std::io::Result<String> {
    Ok(std::fs::read_to_string(path)?.trim().to_string())
}
