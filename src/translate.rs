//! Responses API <-> Chat Completions translation, for providers (like
//! Z.ai/GLM) that only speak Chat Completions. Codex only ever builds
//! Responses-shaped requests, so this bridges both directions.
//!
//! Streaming is handled by always calling upstream non-streaming, then
//! replaying the complete result as a single-shot synthesized SSE sequence.
//! That trades live token-by-token output for a much smaller, lower-risk
//! translation surface (no incremental delta bookkeeping to get subtly
//! wrong).

use crate::config::last_error_file;
use crate::keyfile::load_key;
use crate::logging::log;
use crate::routes::Route;
use crate::schema::sanitize_tools_for_route;
use crate::status::from_u16;
use actix_web::HttpResponse;
use serde_json::{Value, json};

/// GLM-5.3 (and its Flash variant) requires `thinking.type = "enabled"` on
/// every request — disabling thinking is not supported — plus a
/// `reasoning_effort` of `low`/`high`/`max` (no `medium` tier, unlike the
/// Responses API's own effort scale). `max` is Z.ai's documented
/// recommendation for coding tasks; this only rounds Codex's *requested*
/// effort onto GLM's coarser scale, it never overrides it.
fn glm53_thinking_fields(model: &str) -> bool {
    model.starts_with("glm-5.3")
}

fn map_reasoning_effort(effort: &str) -> &'static str {
    match effort {
        "none" | "minimal" | "low" => "low",
        "medium" | "high" => "high",
        _ => "max", // "xhigh", "max", "ultra", and anything unrecognized.
    }
}

fn extract_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .map(|part| match part {
                Value::String(s) => s.clone(),
                _ => part
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            })
            .collect(),
        _ => String::new(),
    }
}

/// Converts a Responses-API request body into a Chat Completions request
/// body. Reasoning items and any input type with no Chat Completions
/// equivalent are dropped rather than erroring — Codex tolerates a reply
/// that simply omits them.
pub fn responses_to_chat_body(parsed: &Value) -> Value {
    let mut messages: Vec<Value> = Vec::new();

    if let Some(instructions) = parsed.get("instructions").and_then(Value::as_str)
        && !instructions.is_empty()
    {
        messages.push(json!({ "role": "system", "content": instructions }));
    }

    let mut pending_tool_call_idx: Option<usize> = None;
    if let Some(Value::Array(input)) = parsed.get("input") {
        for item in input {
            match item.get("type").and_then(Value::as_str).unwrap_or("") {
                "message" => {
                    let role = match item.get("role").and_then(Value::as_str) {
                        Some("developer") => "system",
                        Some(r) => r,
                        None => "user",
                    };
                    let content = item.get("content").map(extract_text).unwrap_or_default();
                    messages.push(json!({ "role": role, "content": content }));
                    pending_tool_call_idx = None;
                }
                "function_call" => {
                    let call = json!({
                        "id": item.get("call_id").or_else(|| item.get("id")).and_then(Value::as_str).unwrap_or(""),
                        "type": "function",
                        "function": {
                            "name": item.get("name").and_then(Value::as_str).unwrap_or(""),
                            "arguments": item.get("arguments").and_then(Value::as_str).unwrap_or("{}"),
                        }
                    });
                    if let Some(idx) = pending_tool_call_idx {
                        match messages[idx]["tool_calls"].as_array_mut() {
                            Some(calls) => calls.push(call),
                            // The pending message was built by this same
                            // branch below, so `tool_calls` is always an
                            // array — but if that invariant is ever
                            // violated, degrade to a new message instead of
                            // panicking on a live request.
                            None => {
                                messages.push(json!({ "role": "assistant", "content": Value::Null, "tool_calls": [call] }));
                                pending_tool_call_idx = Some(messages.len() - 1);
                            }
                        }
                    } else {
                        messages.push(json!({ "role": "assistant", "content": Value::Null, "tool_calls": [call] }));
                        pending_tool_call_idx = Some(messages.len() - 1);
                    }
                }
                "function_call_output" => {
                    let output = item.get("output").cloned().unwrap_or(Value::Null);
                    let content = match &output {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": item.get("call_id").and_then(Value::as_str).unwrap_or(""),
                        "content": content,
                    }));
                    pending_tool_call_idx = None;
                }
                // Other item types (e.g. reasoning) have no Chat Completions
                // equivalent — dropped.
                _ => {}
            }
        }
    }

    let mut body = json!({
        "model": parsed.get("model").cloned().unwrap_or(Value::Null),
        "messages": messages,
        "stream": false,
    });

    if let Some(Value::Array(tools)) = parsed.get("tools") {
        let mapped: Vec<Value> = tools
            .iter()
            .filter(|t| t.get("type").and_then(Value::as_str) == Some("function"))
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.get("name").cloned().unwrap_or(Value::Null),
                        "description": t.get("description").cloned().unwrap_or(Value::Null),
                        "parameters": t.get("parameters").cloned().unwrap_or(Value::Null),
                    }
                })
            })
            .collect();
        if !mapped.is_empty() {
            body["tools"] = Value::Array(mapped);
        }
    }

    if let Some(tool_choice) = parsed.get("tool_choice") {
        body["tool_choice"] = if tool_choice.get("type").and_then(Value::as_str) == Some("function")
        {
            json!({ "type": "function", "function": { "name": tool_choice.get("name").cloned().unwrap_or(Value::Null) } })
        } else {
            tool_choice.clone()
        };
    }

    for key in ["temperature", "top_p"] {
        if let Some(v) = parsed.get(key)
            && v.is_number()
        {
            body[key] = v.clone();
        }
    }
    if let Some(v) = parsed.get("max_output_tokens")
        && v.is_number()
    {
        body["max_tokens"] = v.clone();
    }
    if let Some(v) = parsed.get("parallel_tool_calls") {
        body["parallel_tool_calls"] = v.clone();
    }

    let model = parsed.get("model").and_then(Value::as_str).unwrap_or("");
    if glm53_thinking_fields(model) {
        let effort = parsed
            .pointer("/reasoning/effort")
            .and_then(Value::as_str)
            .unwrap_or("max");
        body["thinking"] = json!({ "type": "enabled" });
        body["reasoning_effort"] = json!(map_reasoning_effort(effort));
    }

    body
}

/// Converts a Chat Completions reply into a Responses-API `response`
/// object (`status: "completed"`, non-streaming shape). The caller decides
/// whether to return this as-is or replay it through
/// [`synthesize_responses_stream`].
pub fn chat_completion_to_responses_object(chat_json: &Value, requested_model: &str) -> Value {
    let message = chat_json
        .pointer("/choices/0/message")
        .cloned()
        .unwrap_or_else(|| json!({}));

    let mut output: Vec<Value> = Vec::new();

    if let Some(Value::Array(tool_calls)) = message.get("tool_calls") {
        for call in tool_calls {
            let id = call.get("id").and_then(Value::as_str).unwrap_or("");
            output.push(json!({
                "type": "function_call",
                "id": format!("fc_{id}"),
                "call_id": id,
                "name": call.pointer("/function/name").cloned().unwrap_or(Value::Null),
                "arguments": call.pointer("/function/arguments").and_then(Value::as_str).unwrap_or("{}"),
                "status": "completed",
            }));
        }
    }

    if let Some(content) = message.get("content").and_then(Value::as_str)
        && !content.is_empty()
    {
        let id = chat_json.get("id").and_then(Value::as_str).unwrap_or("0");
        output.push(json!({
            "type": "message",
            "id": format!("msg_{id}"),
            "role": "assistant",
            "status": "completed",
            "content": [{ "type": "output_text", "text": content, "annotations": [] }],
        }));
    }

    let usage = chat_json.get("usage").cloned().unwrap_or_else(|| json!({}));
    json!({
        "id": chat_json.get("id").cloned().unwrap_or_else(|| json!(format!("resp_{}", chrono::Utc::now().timestamp_millis()))),
        "object": "response",
        "created_at": chat_json.get("created").cloned().unwrap_or_else(|| json!(chrono::Utc::now().timestamp())),
        "status": "completed",
        "model": requested_model,
        "output": output,
        "usage": {
            "input_tokens": usage.get("prompt_tokens").cloned().unwrap_or(json!(0)),
            "output_tokens": usage.get("completion_tokens").cloned().unwrap_or(json!(0)),
            "total_tokens": usage.get("total_tokens").cloned().unwrap_or(json!(0)),
        }
    })
}

fn sse_frame(event: &str, data: &Value) -> String {
    format!("event: {event}\ndata: {data}\n\n")
}

/// Replays a completed `response` object as a single-shot Responses-API SSE
/// stream: `created` → `in_progress` → one added/delta/done/done sequence
/// per output item → `completed`. Not real token-by-token streaming — see
/// the module-level note on why that trade-off was made deliberately.
pub fn synthesize_responses_stream(resp_obj: &Value) -> String {
    let mut out = String::new();
    let base = json!({
        "id": resp_obj.get("id").cloned().unwrap_or(Value::Null),
        "object": "response",
        "created_at": resp_obj.get("created_at").cloned().unwrap_or(Value::Null),
        "model": resp_obj.get("model").cloned().unwrap_or(Value::Null),
    });

    let mut in_progress = base.clone();
    in_progress["status"] = json!("in_progress");
    in_progress["output"] = json!([]);
    out += &sse_frame(
        "response.created",
        &json!({ "type": "response.created", "response": in_progress }),
    );
    out += &sse_frame(
        "response.in_progress",
        &json!({ "type": "response.in_progress", "response": in_progress }),
    );

    let empty: Vec<Value> = vec![];
    let items = resp_obj
        .get("output")
        .and_then(Value::as_array)
        .unwrap_or(&empty);
    for (index, item) in items.iter().enumerate() {
        match item.get("type").and_then(Value::as_str).unwrap_or("") {
            "message" => {
                let item_id = item.get("id").and_then(Value::as_str).unwrap_or("");
                let text = item
                    .pointer("/content/0/text")
                    .and_then(Value::as_str)
                    .unwrap_or("");

                out += &sse_frame(
                    "response.output_item.added",
                    &json!({
                        "type": "response.output_item.added", "output_index": index,
                        "item": { "id": item_id, "type": "message", "status": "in_progress", "role": "assistant", "content": [] }
                    }),
                );
                out += &sse_frame(
                    "response.content_part.added",
                    &json!({
                        "type": "response.content_part.added", "item_id": item_id, "output_index": index, "content_index": 0,
                        "part": { "type": "output_text", "text": "", "annotations": [] }
                    }),
                );
                out += &sse_frame(
                    "response.output_text.delta",
                    &json!({
                        "type": "response.output_text.delta", "item_id": item_id, "output_index": index, "content_index": 0, "delta": text
                    }),
                );
                out += &sse_frame(
                    "response.output_text.done",
                    &json!({
                        "type": "response.output_text.done", "item_id": item_id, "output_index": index, "content_index": 0, "text": text
                    }),
                );
                out += &sse_frame(
                    "response.content_part.done",
                    &json!({
                        "type": "response.content_part.done", "item_id": item_id, "output_index": index, "content_index": 0,
                        "part": { "type": "output_text", "text": text, "annotations": [] }
                    }),
                );
                out += &sse_frame(
                    "response.output_item.done",
                    &json!({
                        "type": "response.output_item.done", "output_index": index, "item": item
                    }),
                );
            }
            "function_call" => {
                let item_id = item.get("id").and_then(Value::as_str).unwrap_or("");
                let call_id = item.get("call_id").and_then(Value::as_str).unwrap_or("");
                let name = item.get("name").and_then(Value::as_str).unwrap_or("");
                let arguments = item
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("{}");

                out += &sse_frame(
                    "response.output_item.added",
                    &json!({
                        "type": "response.output_item.added", "output_index": index,
                        "item": { "id": item_id, "type": "function_call", "status": "in_progress", "call_id": call_id, "name": name, "arguments": "" }
                    }),
                );
                out += &sse_frame(
                    "response.function_call_arguments.delta",
                    &json!({
                        "type": "response.function_call_arguments.delta", "item_id": item_id, "output_index": index, "delta": arguments
                    }),
                );
                out += &sse_frame(
                    "response.function_call_arguments.done",
                    &json!({
                        "type": "response.function_call_arguments.done", "item_id": item_id, "output_index": index, "arguments": arguments
                    }),
                );
                out += &sse_frame(
                    "response.output_item.done",
                    &json!({
                        "type": "response.output_item.done", "output_index": index, "item": item
                    }),
                );
            }
            _ => {}
        }
    }

    let mut completed = base.clone();
    completed["status"] = json!("completed");
    completed["output"] = resp_obj.get("output").cloned().unwrap_or_else(|| json!([]));
    completed["usage"] = resp_obj.get("usage").cloned().unwrap_or_else(|| json!({}));
    out += &sse_frame(
        "response.completed",
        &json!({ "type": "response.completed", "response": completed }),
    );

    out
}

/// Handles one request for a [`Translate::Chat`](crate::routes::Translate::Chat)
/// route end to end: translate the body, call upstream, translate the
/// reply back, and log the outcome (including writing
/// [`last_error_file`] on a non-2xx upstream response).
pub async fn handle_translated(
    client: &reqwest::Client,
    route: &Route,
    mut parsed: Value,
) -> HttpResponse {
    let wants_stream = parsed
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let model = parsed
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    sanitize_tools_for_route(&mut parsed);
    let chat_body = responses_to_chat_body(&parsed);

    let key = match load_key(&route.key_file) {
        Ok(k) => k,
        Err(e) => {
            log(&format!("error key {:?}: {e}", route.key_file));
            return HttpResponse::InternalServerError()
                .content_type("text/plain")
                .body(format!("codex-router: cannot read {} key", route.name));
        }
    };

    let url = format!("{}{}", route.base_url, route.path);
    let sent = client
        .post(&url)
        .bearer_auth(key)
        .json(&chat_body)
        .send()
        .await;

    let response = match sent {
        Ok(r) => r,
        Err(e) => {
            log(&format!("error {} POST: {e}", route.name));
            return HttpResponse::BadGateway()
                .content_type("text/plain")
                .body("codex-router: upstream error");
        }
    };

    let status = response.status();
    let raw = response.text().await.unwrap_or_default();
    log(&format!(
        "{} POST model={model} -> {}",
        route.name,
        status.as_u16()
    ));

    if !status.is_success() {
        let _ = std::fs::write(
            last_error_file(),
            serde_json::to_vec_pretty(&json!({
                "route": route.name, "model": model, "status": status.as_u16(), "body": raw,
            }))
            .unwrap_or_default(),
        );
        return HttpResponse::build(from_u16(status.as_u16()))
            .content_type("application/json")
            .body(raw);
    }

    let chat_json: Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            log(&format!(
                "error parsing upstream JSON for {}: {e}",
                route.name
            ));
            return HttpResponse::BadGateway()
                .content_type("text/plain")
                .body("codex-router: invalid upstream response");
        }
    };

    let resp_obj = chat_completion_to_responses_object(&chat_json, &model);

    if wants_stream {
        HttpResponse::Ok()
            .content_type("text/event-stream")
            .append_header(("cache-control", "no-cache"))
            .body(synthesize_responses_stream(&resp_obj))
    } else {
        HttpResponse::Ok()
            .content_type("application/json")
            .json(resp_obj)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translates_plain_text_request() {
        let parsed = json!({
            "model": "glm-5.3-flash",
            "instructions": "You are helpful.",
            "input": [{ "type": "message", "role": "user", "content": [{ "type": "input_text", "text": "hi" }] }],
        });
        let body = responses_to_chat_body(&parsed);
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][1]["role"], "user");
        assert_eq!(body["messages"][1]["content"], "hi");
        assert_eq!(body["stream"], false);
    }

    #[test]
    fn translates_tool_round_trip() {
        let parsed = json!({
            "model": "glm-5.3-flash",
            "input": [
                { "type": "message", "role": "user", "content": [{ "type": "input_text", "text": "weather?" }] },
                { "type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{\"city\":\"Harare\"}" },
                { "type": "function_call_output", "call_id": "call_1", "output": "22C sunny" },
            ],
        });
        let body = responses_to_chat_body(&parsed);
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(
            messages[1]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        assert_eq!(messages[2]["role"], "tool");
        assert_eq!(messages[2]["tool_call_id"], "call_1");
    }

    #[test]
    fn injects_thinking_fields_for_glm53_family() {
        for model in ["glm-5.3", "glm-5.3-flash"] {
            let parsed = json!({
                "model": model,
                "input": [{ "type": "message", "role": "user", "content": [{ "type": "input_text", "text": "hi" }] }],
                "reasoning": { "effort": "medium" },
            });
            let body = responses_to_chat_body(&parsed);
            assert_eq!(
                body["thinking"],
                json!({ "type": "enabled" }),
                "model={model}"
            );
            assert_eq!(body["reasoning_effort"], "high", "model={model}");
        }
    }

    #[test]
    fn omits_thinking_fields_for_non_glm53_models() {
        let parsed = json!({
            "model": "glm-4.5-flash",
            "input": [{ "type": "message", "role": "user", "content": [{ "type": "input_text", "text": "hi" }] }],
        });
        let body = responses_to_chat_body(&parsed);
        assert!(body.get("thinking").is_none());
        assert!(body.get("reasoning_effort").is_none());
    }

    #[test]
    fn defaults_reasoning_effort_to_max_when_unspecified() {
        let parsed = json!({
            "model": "glm-5.3-flash",
            "input": [{ "type": "message", "role": "user", "content": [{ "type": "input_text", "text": "hi" }] }],
        });
        let body = responses_to_chat_body(&parsed);
        assert_eq!(body["reasoning_effort"], "max");
    }

    #[test]
    fn maps_every_responses_effort_onto_glms_three_tiers() {
        for (input, expected) in [
            ("none", "low"),
            ("minimal", "low"),
            ("low", "low"),
            ("medium", "high"),
            ("high", "high"),
            ("xhigh", "max"),
            ("max", "max"),
            ("ultra", "max"),
        ] {
            assert_eq!(map_reasoning_effort(input), expected, "input={input}");
        }
    }

    #[test]
    fn translates_chat_completion_reply_with_tool_call() {
        let chat_json = json!({
            "id": "abc",
            "choices": [{ "message": { "tool_calls": [
                { "id": "call_9", "type": "function", "function": { "name": "get_weather", "arguments": "{}" } }
            ] } }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15 },
        });
        let resp = chat_completion_to_responses_object(&chat_json, "glm-5.3-flash");
        assert_eq!(resp["output"][0]["type"], "function_call");
        assert_eq!(resp["output"][0]["call_id"], "call_9");
        assert_eq!(resp["usage"]["input_tokens"], 10);
    }

    #[test]
    fn synthesized_stream_contains_expected_events() {
        let resp_obj = json!({
            "id": "r1", "created_at": 1, "model": "glm-5.3-flash",
            "output": [{ "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                          "content": [{ "type": "output_text", "text": "ok", "annotations": [] }] }],
            "usage": { "input_tokens": 1, "output_tokens": 1, "total_tokens": 2 },
        });
        let sse = synthesize_responses_stream(&resp_obj);
        assert!(sse.contains("event: response.created"));
        assert!(sse.contains("event: response.output_text.delta"));
        assert!(sse.contains("\"delta\":\"ok\""));
        assert!(sse.contains("event: response.completed"));
    }
}
