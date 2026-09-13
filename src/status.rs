//! Safe conversion from an upstream numeric status code to Actix's status
//! type — isolated here so no call site needs to reach for `unwrap`.

use actix_web::http::StatusCode;

/// Converts a `u16` status code to Actix's `StatusCode`, falling back to
/// 502 Bad Gateway if upstream ever returns something outside the valid
/// range. `reqwest::StatusCode` is always in range in practice, but this
/// keeps a malformed upstream from panicking a live request.
pub fn from_u16(code: u16) -> StatusCode {
    StatusCode::from_u16(code).unwrap_or(StatusCode::BAD_GATEWAY)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_code_round_trips() {
        assert_eq!(from_u16(404), StatusCode::NOT_FOUND);
    }

    #[test]
    fn out_of_range_falls_back_to_bad_gateway() {
        assert_eq!(from_u16(9999), StatusCode::BAD_GATEWAY);
    }
}
