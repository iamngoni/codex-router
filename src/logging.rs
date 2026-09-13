//! Minimal append-and-echo request logging.
//!
//! Owns only formatting and best-effort persistence — a failed disk write
//! never blocks or fails a request; the line still reaches stdout, which
//! `launchd` captures separately as a fallback.

use crate::config::log_file;
use chrono::SecondsFormat;
use std::fs::OpenOptions;
use std::io::Write;

/// Appends a timestamped line to the log file and stdout. Never panics: a
/// disk write failure is silently swallowed rather than taking down a
/// request that otherwise succeeded.
pub fn log(message: &str) {
    let line = format!(
        "{} {}\n",
        chrono::Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true),
        message
    );
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_file())
    {
        let _ = file.write_all(line.as_bytes());
    }
    print!("{line}");
}
