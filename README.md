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
| `openrouter/` | OpenRouter | Responses API | pass-through (prefix stripped, tool-schema sanitized) |
| `glm-` | Z.ai (GLM) | Chat Completions | translated both ways |
| anything else | OpenAI | Responses API | pass-through, unmodified |

Each provider's API key lives in `~/.config/<provider>/key` (mode 600),
outside this repository.

## Anthropic Messages compatibility

One native Anthropic Messages handler serves all of the stock Claude client
paths:

```text
POST /backend-api/claude
POST /backend-api/claude/v1/messages
POST /v1/messages
```

`claude-*` models go directly to `https://api.anthropic.com/v1/messages` and
use the caller's `Authorization: Bearer ...` or `x-api-key` credential. The
`anthropic-version` and `anthropic-beta` headers, query strings such as
`?beta=true`, content blocks, and SSE frames are preserved. Claude models do
not fall back to OpenAI/Codex.

Native Messages provider routes are selected by model prefix:

| Model prefix | Upstream Messages endpoint | Credential |
|---|---|---|
| `deepseek-` | `https://api.deepseek.com/anthropic/v1/messages` | `~/.config/deepseek/key` |
| `openrouter/` | `https://openrouter.ai/api/v1/messages` (prefix removed from `model`) | `~/.config/openrouter/key` |
| `glm-` | `https://api.z.ai/api/anthropic/v1/messages` | `~/.config/zai/key` |
| `muse-` | unsupported until native Messages behavior is verified | — |

For third-party routes, caller `Authorization`, `x-api-key`, and cookies are
never forwarded; the configured route key is sent as `Authorization: Bearer`.
Successful and error responses are streamed back with their upstream status,
safe headers, and body bytes intact.

OpenRouter's model ids are namespaced by vendor, so the `openrouter/` prefix is
Codex's naming and only Codex's: `openrouter/anthropic/claude-sonnet-4.5` is
forwarded to `https://openrouter.ai/api/v1/responses` as
`anthropic/claude-sonnet-4.5`, and the request log says so
(`model=openrouter/… forwarded=anthropic/…`). Its key is
`~/.config/openrouter/key`. No body-size limit has been measured for it, so the
[oversized-request handling](#oversized-requests) below does not apply — big
requests go straight to OpenRouter and it decides.

One wire difference is corrected on the way back: OpenRouter streams Responses
events as bare `data:` frames, while OpenAI and DeepSeek put an
`event: <type>` line in front of each one and Codex was written against that.
`Route::label_sse_events` marks the routes that need the label, and
`proxy::label_sse_events` adds it from the payload's own `type` field. Frames
without one — keep-alive comments, `data: [DONE]` — pass through untouched, as
does a stream that arrives already labelled.

Codex only offers models it has catalog entries for, and the slug is the string
it sends as `model`, so OpenRouter entries have to be named
`openrouter/<vendor>/<model>`. `GET /backend-api/codex/models` fetches the native
OpenAI catalogue on every request, then merges entries from
`~/.codex/model-catalogs/external.json`. Native records win slug collisions;
external records keep their file order and receive priorities after the native
catalogue. A malformed or unavailable external file leaves the last valid file
in place, while a valid empty `models` list clears the external entries.

`tools/openrouter-catalog.py` generates external entries from OpenRouter's public
model list (context window, modalities, whether the model takes reasoning) and
atomically replaces only existing `openrouter/` entries, preserving manual
DeepSeek/GLM entries:

```sh
python3 tools/openrouter-catalog.py                        # preview
python3 tools/openrouter-catalog.py --keep-variants --install   # all tool-capable models
python3 tools/openrouter-catalog.py --all --install             # everything the API lists
```

## Oversized requests

DeepSeek's edge (openresty in front of `api.deepseek.com`) rejects request
bodies larger than 48 MiB (50,331,648 bytes — measured against the live edge,
not taken from docs) with a bare HTML `413` that names no limit. Codex retries
a failing turn a few times, so one oversized turn means six uploads of tens of
megabytes and still no answer. The usual cause is a forked thread replaying a
long history — screenshots and tool output dominate — or one oversized
attachment.

`routes::DEEPSEEK_MAX_BODY_BYTES` records that limit, and an over-limit
`deepseek-` request is made to fit before it is forwarded (`trim.rs`), cheapest
to lose first:

1. `encrypted_content` on reasoning items — opaque blobs minted by another
   provider, unreadable to anyone else — are dropped.
2. Attachments in *history* (any base64 image older than the message being
   answered) are replaced by a marker, oldest first, and only as many as the
   overflow needs.
3. Tool outputs are cut to a common level, largest first, each keeping its
   opening and closing bytes around a marker. Nothing goes below 2 KiB, and
   pasted attachments that arrived as message text (`≥ 16 KiB` of it) are cut
   the same way.
4. Only if none of that is enough, an attachment in the message being answered
   goes too.

The item graph itself is never touched: nothing is added, removed, or reordered,
so `function_call` ↔ `function_call_output` pairing, ids and item types survive,
and the message the user is asking about is never rewritten. Each trim logs
`trimmed route=… saved … (…)`, and every cut leaves a marker naming what went
missing, so the model can tell the user what it can no longer see.

If those three steps cannot get the body under the cap, the request is refused
locally, before the provider is called, with a Responses-shaped 413 naming both
sizes:

```json
{"error": {"message": "codex-router: request body is 50.7 MiB, over deepseek's 48.0 MiB. …", "type": "invalid_request_error", "code": "payload_too_large"}}
```

Routes with no measured limit (`glm-`, `muse-`, OpenAI's own models) are
forwarded unchecked.

This guard is about the edge's *byte* limit, not the model's context window.
DeepSeek answers an over-window request with a normal JSON `400` (`This model's
maximum context length is 1048576 tokens…`), which Codex can act on — a 48 MiB
body of pure text is millions of tokens. A trimmed request only fits *and*
succeeds when the bytes were dominated by images or opaque blobs, which is what
these oversized requests usually are.

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

```sh
cargo test --test live_openrouter -- --ignored
```

Same deal for OpenRouter, and it covers the two things specific to that route:
the `openrouter/` prefix comes off before forwarding, and a streamed reply comes
back labelled the way Codex expects it. Costs a fraction of a cent on
`openai/gpt-5-nano`; needs a real `~/.config/openrouter/key`.

Native Messages probes are also opt-in:

```sh
cargo test --test live_messages -- --ignored
```

They call DeepSeek (`deepseek-flash`), Z.ai (`glm-5.3-flash`), and OpenRouter
(`openrouter/openai/gpt-5-nano`) through the Messages handler. They require the
corresponding key files, spend provider credits, and are not run by default.

### Coverage

```sh
cargo llvm-cov --all-features --workspace --ignore-filename-regex 'main\.rs$'
```

`main.rs` is excluded as the one narrow, documented exception: it's pure
process bootstrap (`.bind()?.run().await`), nothing to unit test. With that
exclusion, line coverage is **~90.5%**, short of the 95% bar. The remaining
gap is concentrated in provider-error branches, Messages streaming failures,
and the live-only dispatch paths. The opt-in tests prove those real provider
routes, but `#[ignore]`d network tests do not count toward the automated
number. Closing the gap fully would mean making more of the routing and
transport layer injectable purely for testability — a deliberate scope call
left as-is rather than done silently.
