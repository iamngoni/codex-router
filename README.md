# codex-router

Local reverse proxy that lets [Codex](https://github.com/openai/codex) route
different model-name prefixes to different providers, translating
Responses API ↔ Chat Completions where a provider only speaks the latter.

Codex's active `model_provider` is always `router`
(`http://127.0.0.1:4141/backend-api/codex`); this process inspects the
`model` field of each request and dispatches accordingly.

## Routes

| Prefix | Provider | Shape | Handling |
|---|---|---|---|
| `deepseek-` | DeepSeek | Responses API | pass-through (tool-schema sanitized) |
| `muse-` | Meta Muse | Responses API | pass-through (tool-schema sanitized) |
| `glm-` | Z.ai (GLM) | Chat Completions | translated both ways |
| anything else | OpenAI | Responses API | pass-through, unmodified |

Each provider's API key lives in `~/.config/<provider>/key` (mode 600),
outside this repository.

## GLM translation

Z.ai has no Responses-API endpoint — only `/chat/completions`. Codex only
ever builds Responses-shaped requests, so the `glm-` route:

1. Converts the Responses body (`input`, `instructions`, `tools`,
   `tool_choice`, tool-call round trips) into a Chat Completions body.
2. Calls Z.ai non-streaming, regardless of what Codex asked for.
3. Converts the reply back into a Responses `response` object.
4. If Codex requested streaming, **replays the complete reply as a
   single-shot synthesized SSE sequence** (`response.created` →
   `response.output_item.added` → one delta → `done` → `response.completed`)
   rather than streaming real per-token deltas from Z.ai. This trades live
   token-by-token output for a much smaller, lower-risk translation surface.

GLM-5.3 (and `-flash`) additionally requires `thinking.type = "enabled"` on
every request (disabling thinking is unsupported) plus a `reasoning_effort`
of `low`/`high`/`max` — narrower than the Responses API's own effort scale.
`responses_to_chat_body` rounds Codex's requested effort onto that scale;
see `translate::map_reasoning_effort`.

## Running

```sh
cargo build --release
./target/release/codex-router
```

Listens on `127.0.0.1:4141`. `GET /healthz` returns `200 ok`. Logs to
`~/.local/state/codex-router.log`; the most recent upstream error (status
≥ 400) is written to `~/.local/state/codex-router-last-error.json`.

In production this runs under a `launchd` LaunchAgent
(`~/Library/LaunchAgents/com.modestnerd.codex-router.plist`) pointed at the
release binary, `RunAtLoad` + `KeepAlive`.

## Testing

```sh
cargo test
```

Unit tests cover the pure translation/schema logic. `tests/http.rs` mocks
the upstream with `wiremock` to exercise `handle_translated` and
`handle_passthrough` end to end: plain text, streaming, tool calls,
auth-header replacement, missing-key/connect-failure/malformed-upstream-JSON
errors, and the `>=400` last-error-file write (the two tests covering that
last one deliberately overwrite the real `~/.local/state/
codex-router-last-error.json` — there's no test-only override for that
path, and it's a disposable scratch file the running service already
overwrites on every real failure, so a synthetic entry from `cargo test` is
an accepted trade-off for actually covering that write).

```sh
cargo test --test live_glm -- --ignored
```

Opt-in, hits the real Z.ai API through the full `dispatch` routing glue
(the one path the hermetic suite can't reach, since `find_route` always
resolves to the real provider hosts). Requires a real `~/.config/zai/key`
and spends real credits — not run by default.

### Coverage

```sh
cargo llvm-cov --all-features --workspace --ignore-filename-regex 'main\.rs$'
```

`main.rs` is excluded as the one narrow, documented exception: it's pure
process bootstrap (`.bind()?.run().await`), nothing to unit test. With that
exclusion, line coverage is **~84.6%**, short of the 95% bar. The gap is
almost entirely `lib.rs`'s `dispatch` — its constituent logic (`find_route`,
`handle_translated`, `handle_passthrough`) is fully covered individually,
and the whole path is proven correct by `tests/live_glm.rs`, but that test
is `#[ignore]`d (real network/credentials) so it doesn't count toward the
automated number. Closing this fully would mean making the routing table
injectable purely for testability — a deliberate scope call left as-is
rather than done silently.
