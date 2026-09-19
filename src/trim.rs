//! Trims an oversized Responses request down to a size the provider's edge
//! accepts, instead of letting the turn die at the transport layer.
//!
//! Same spirit as [`crate::schema`], which rewrites tool schemas a provider
//! cannot parse: the request is adjusted so it can be answered at all. The rule
//! here is narrower — **the item graph never changes**. Nothing is added,
//! removed, or reordered, so `function_call` ↔ `function_call_output` pairing,
//! ids and item types stay exactly as Codex built them. Only string payloads
//! *inside* an existing item are replaced, cheapest-to-lose first:
//!
//! 1. `encrypted_content` on reasoning items — opaque blobs minted by another
//!    provider (Codex asks for them via `include`). They are megabytes and
//!    unreadable to anyone else, so they go first.
//! 2. Attachments from *history* — base64 images older than the message being
//!    answered — are replaced by a one-line marker, oldest first, and only as
//!    many as the overflow needs. They are what makes these requests huge, and
//!    the user has already moved on from them.
//! 3. Tool outputs (`function_call_output` / `custom_tool_call_output`) are cut
//!    to a common level, so the largest lose most and small ones are left
//!    alone. Each keeps its opening and closing bytes around a marker naming how
//!    much was removed, and none goes below [`OUTPUT_FLOOR_BYTES`]. Pasted
//!    attachments that arrived as message text are treated the same way.
//! 4. Only if none of that is enough, an attachment in the message being
//!    answered is dropped too.
//!
//! Two things are never touched: the message the user is asking about, and short
//! message text — instructions and replies are the conversation itself, and
//! silently mangling those would be worse than failing. If the steps above
//! cannot bring the body under the cap, [`fit_to_cap`] gives up (`None`) and the
//! caller refuses the request.

use serde_json::{Value, json};

/// Never shrink a tool output below this. The marker plus a readable excerpt is
/// still worth sending; below it, we would rather refuse the request than
/// pretend the model can work from a stub.
const OUTPUT_FLOOR_BYTES: usize = 2 * 1024;

/// Share of a truncated payload that keeps the opening bytes; the rest keeps
/// the closing bytes, where command output puts its errors and summaries.
const OUTPUT_HEAD_SHARE: usize = 60;

/// Smallest message text that counts as a pasted attachment rather than the
/// conversation. Trimming the middle of a 200 KB paste is fair; trimming what
/// someone typed is not.
const ATTACHMENT_MIN_BYTES: usize = 16 * 1024;

/// Roughly what one inserted marker costs, in bytes — the excerpt that replaces
/// a cut payload grows by this much before it saves anything.
const MARKER_BYTES: usize = 128;

/// How many passes a trim may take. One is enough whenever the estimates hold;
/// the retries exist because JSON escaping and the inserted markers make sizes a
/// little unpredictable, and every retry removes only what is still over.
const TRIM_PASSES: usize = 4;

/// What a trim attempt came to.
#[derive(Debug, PartialEq, Eq)]
pub enum Fitted {
    /// Already under the cap; nothing was touched.
    Untouched,
    /// Under the cap now; this is what was removed.
    Trimmed(TrimReport),
    /// Still over the cap — this is what was removed trying.
    CannotFit(TrimReport),
}

/// What one trim removed.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct TrimReport {
    /// Serialized bytes between the original body and the trimmed one.
    pub bytes_saved: usize,
    pub truncated_outputs: usize,
    /// Pasted attachments that arrived as message text and were cut down.
    pub trimmed_message_parts: usize,
    pub removed_images: usize,
    pub stripped_reasoning_blobs: usize,
}

impl TrimReport {
    /// What was removed, for an error message — e.g. `2 images, 41 tool outputs`.
    pub fn clause(&self) -> String {
        let counted = [
            (self.removed_images, "image"),
            (self.truncated_outputs, "tool output"),
            (self.trimmed_message_parts, "pasted attachment"),
            (self.stripped_reasoning_blobs, "reasoning blob"),
        ];
        let parts: Vec<String> = counted
            .iter()
            .filter(|(count, _)| *count > 0)
            .map(|(count, noun)| match count {
                1 => format!("1 {noun}"),
                n => format!("{n} {noun}s"),
            })
            .collect();
        if parts.is_empty() {
            return "nothing".to_string();
        }
        parts.join(", ")
    }

    /// One log-line's worth of what happened.
    pub fn summary(&self) -> String {
        format!(
            "saved {} ({} tool outputs cut, {} pasted attachments cut, {} images dropped, {} reasoning blobs stripped)",
            human_bytes(self.bytes_saved),
            self.truncated_outputs,
            self.trimmed_message_parts,
            self.removed_images,
            self.stripped_reasoning_blobs
        )
    }
}

/// Shrinks `body` until its serialized form fits `cap`, reporting what was
/// removed. A [`Fitted::CannotFit`] leaves `body` partially trimmed; callers
/// refuse the request in that case, so the body is discarded either way.
pub fn fit_to_cap(body: &mut Value, cap: usize) -> Fitted {
    let original = serialized_len(body);
    if original <= cap {
        return Fitted::Untouched;
    }

    let mut report = TrimReport::default();
    // Which payloads have already been cut: passes repeat, and the report counts
    // saved *outputs*, not how many cuts it took to get there.
    let mut cut: Vec<(usize, (u8, usize))> = Vec::new();
    strip_reasoning_blobs(body, &mut report);

    // Each layer pays only its share of the overflow, cheapest to lose first:
    // history's attachments, then the text around them, and the message being
    // answered only if there is nothing else left.
    for _ in 0..TRIM_PASSES {
        let size = serialized_len(body);
        if size <= cap {
            break;
        }
        // A little slack, since JSON escaping and the markers themselves are a
        // few percent of what a pass removes.
        remove_images(body, size - cap + size / 100, &mut report, Scope::History);

        let size = serialized_len(body);
        if size <= cap {
            break;
        }
        truncate_outputs(body, size - cap + size / 100, &mut report, &mut cut);

        let size = serialized_len(body);
        if size <= cap {
            break;
        }
        remove_images(body, size - cap + size / 100, &mut report, Scope::LatestAsk);
    }

    let size = serialized_len(body);
    report.bytes_saved = original - size;
    if size > cap {
        return Fitted::CannotFit(report);
    }
    Fitted::Trimmed(report)
}

/// Serialized size of a body, in bytes.
pub fn serialized_len(value: &Value) -> usize {
    serde_json::to_vec(value).map(|v| v.len()).unwrap_or(0)
}

/// Sizes as an operator reads them in a terminal.
pub fn human_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

/// Drops `encrypted_content` from reasoning items — provider-minted blobs that
/// no other provider can read.
fn strip_reasoning_blobs(body: &mut Value, report: &mut TrimReport) {
    let Some(items) = items_mut(body) else { return };
    for item in items {
        let Some(obj) = item.as_object_mut() else {
            continue;
        };
        if let Some(Value::String(blob)) = obj.remove("encrypted_content")
            && !blob.is_empty()
        {
            report.stripped_reasoning_blobs += 1;
        }
    }
}

/// Which items an image sweep may touch.
#[derive(Clone, Copy)]
enum Scope {
    /// Everything older than the message being answered — attachments the user
    /// has already moved on from.
    History,
    /// That message itself, kept whole until nothing else can pay.
    LatestAsk,
}

/// Replaces base64 image payloads with a marker, oldest first, and only as many
/// as `needed` bytes require.
fn remove_images(body: &mut Value, needed: usize, report: &mut TrimReport, scope: Scope) {
    let targets: Vec<usize> = {
        let Some(items) = items_ref(body) else { return };
        let ask = ask_index(items);
        items
            .iter()
            .enumerate()
            .filter(|(index, _)| match scope {
                Scope::History => Some(*index) != ask,
                Scope::LatestAsk => Some(*index) == ask,
            })
            .map(|(index, _)| index)
            .collect()
    };

    let Some(items) = items_mut(body) else { return };
    let mut saved = 0;
    for index in targets {
        for field in ["output", "content"] {
            let Some(parts) = items[index].get_mut(field).and_then(Value::as_array_mut) else {
                continue;
            };
            for part in parts.iter_mut() {
                if saved >= needed {
                    return;
                }
                let Some(marker) = image_marker(part) else {
                    continue;
                };
                let before = serialized_len(part);
                *part = json!({ "type": "input_text", "text": marker });
                saved += before.saturating_sub(serialized_len(part));
                report.removed_images += 1;
            }
        }
    }
}

/// Index of the message the user is currently asking about — the newest user
/// message in the request, wherever it sits among the tool calls that follow it.
fn ask_index(items: &[Value]) -> Option<usize> {
    items
        .iter()
        .rposition(|item| item.get("role").and_then(Value::as_str) == Some("user"))
}

/// Replacement text for an image content part carrying a base64 payload, or
/// `None` for every other part.
fn image_marker(part: &Value) -> Option<String> {
    let obj = part.as_object()?;
    if !matches!(
        obj.get("type").and_then(Value::as_str),
        Some("input_image") | Some("image_url")
    ) {
        return None;
    }
    let payload = obj
        .get("image_url")
        .and_then(|u| u.as_str().or_else(|| u.get("url").and_then(Value::as_str)))?;
    Some(format!(
        "[codex-router: image removed — {} of image data did not fit the provider's request size limit]",
        human_bytes(payload.len())
    ))
}

/// A string inside a tool-output item that can be replaced without touching the
/// item's structure.
#[derive(Clone, Copy)]
enum Slot {
    /// `output` holding one string.
    Output,
    /// `output[index].text` — tool results that carry content parts.
    OutputPart(usize),
    /// `content[index].text`.
    ContentPart(usize),
}

impl Slot {
    /// Identity of this slot inside its item, for de-duplicating cut counts.
    fn key(self) -> (u8, usize) {
        match self {
            Slot::Output => (0, 0),
            Slot::OutputPart(index) => (1, index),
            Slot::ContentPart(index) => (2, index),
        }
    }
}

fn is_tool_output(item: &Value) -> bool {
    matches!(
        item.get("type").and_then(Value::as_str),
        Some("function_call_output") | Some("custom_tool_call_output")
    )
}

/// Payloads worth cutting inside one item. A tool output is machinery, so it
/// keeps its ends at any size; a message's text is only touched once it is big
/// enough to be a pasted attachment rather than something someone typed.
fn trimmable_slots(item: &Value) -> Vec<(Slot, usize)> {
    if is_tool_output(item) {
        return slots(item);
    }
    slots(item)
        .into_iter()
        .filter(|(_, len)| *len >= ATTACHMENT_MIN_BYTES)
        .collect()
}

/// String payloads of a tool-output item with their current sizes.
fn slots(item: &Value) -> Vec<(Slot, usize)> {
    let mut found = Vec::new();
    match item.get("output") {
        Some(Value::String(output)) => found.push((Slot::Output, output.len())),
        Some(Value::Array(parts)) => {
            for (index, part) in parts.iter().enumerate() {
                if let Some(Value::String(text)) = part.get("text") {
                    found.push((Slot::OutputPart(index), text.len()));
                }
            }
        }
        _ => {}
    }
    if let Some(parts) = item.get("content").and_then(Value::as_array) {
        for (index, part) in parts.iter().enumerate() {
            if let Some(Value::String(text)) = part.get("text") {
                found.push((Slot::ContentPart(index), text.len()));
            }
        }
    }
    found
}

fn slot_mut(item: &mut Value, slot: Slot) -> Option<&mut String> {
    let value = match slot {
        Slot::Output => item.get_mut("output")?,
        Slot::OutputPart(index) => item.get_mut("output")?.get_mut(index)?.get_mut("text")?,
        Slot::ContentPart(index) => item.get_mut("content")?.get_mut(index)?.get_mut("text")?,
    };
    match value {
        Value::String(text) => Some(text),
        _ => None,
    }
}

/// Cuts tool outputs down to a common level, removing at least `save_at_least`
/// bytes but no more than that: payloads already below the level survive
/// untouched, and no payload goes below [`OUTPUT_FLOOR_BYTES`].
fn truncate_outputs(
    body: &mut Value,
    save_at_least: usize,
    report: &mut TrimReport,
    cut: &mut Vec<(usize, (u8, usize))>,
) {
    let found: Vec<(usize, Slot, usize)> = {
        let Some(items) = items_ref(body) else { return };
        let ask = ask_index(items);
        items
            .iter()
            .enumerate()
            // The message being answered stays whole; everything around it pays.
            .filter(|(index, _)| Some(*index) != ask)
            .flat_map(|(index, item)| {
                trimmable_slots(item)
                    .into_iter()
                    .map(move |(slot, len)| (index, slot, len))
            })
            .collect()
    };
    if found.is_empty() {
        return;
    }

    let lens: Vec<usize> = found.iter().map(|(_, _, len)| *len).collect();
    // Every cut leaves a marker behind, so the raw cut has to be that much
    // bigger than the saving it is meant to buy — otherwise a pass can end up
    // larger than it started.
    let raw_cut = save_at_least.saturating_add(found.len() * MARKER_BYTES);
    let level = level_for_saving(&lens, raw_cut);

    let Some(items) = items_mut(body) else { return };
    for (index, slot, len) in found {
        if len <= level {
            continue;
        }
        let machinery = is_tool_output(&items[index]);
        if let Some(text) = slot_mut(&mut items[index], slot) {
            *text = excerpt(text, level);
            let key = (index, slot.key());
            if !cut.contains(&key) {
                cut.push(key);
                if machinery {
                    report.truncated_outputs += 1;
                } else {
                    report.trimmed_message_parts += 1;
                }
            }
        }
    }
}

/// The highest level — that is, the least cutting — at which the payloads above
/// it give up at least `save_at_least` bytes. Returns `OUTPUT_FLOOR_BYTES` when
/// even cutting everything to the floor cannot free that much.
fn level_for_saving(lens: &[usize], save_at_least: usize) -> usize {
    let highest = lens.iter().copied().max().unwrap_or(0);
    if save_at_least == 0 || highest <= OUTPUT_FLOOR_BYTES {
        return highest.max(OUTPUT_FLOOR_BYTES);
    }
    let (mut low, mut high) = (OUTPUT_FLOOR_BYTES, highest);
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        let saving: usize = lens.iter().map(|len| len.saturating_sub(mid)).sum();
        if saving >= save_at_least {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    low
}

/// `keep` bytes of `text` — its opening [`OUTPUT_HEAD_SHARE`] percent and the
/// remaining share from the end — around a marker naming how much was removed.
/// Cuts land on character boundaries.
fn excerpt(text: &str, keep: usize) -> String {
    let keep = keep.min(text.len());
    let head = floor_boundary(text, keep * OUTPUT_HEAD_SHARE / 100);
    let tail = ceil_boundary(text, text.len() - (keep - head.min(keep)));
    format!(
        "{}\n\n[codex-router: {} removed from this tool output to fit the provider's request size limit]\n\n{}",
        &text[..head],
        human_bytes(tail.saturating_sub(head)),
        &text[tail..]
    )
}

fn floor_boundary(text: &str, mut index: usize) -> usize {
    if index >= text.len() {
        return text.len();
    }
    while !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

fn ceil_boundary(text: &str, mut index: usize) -> usize {
    if index >= text.len() {
        return text.len();
    }
    while !text.is_char_boundary(index) {
        index += 1;
    }
    index
}

fn items_ref(body: &Value) -> Option<&Vec<Value>> {
    body.get("input").and_then(Value::as_array)
}

fn items_mut(body: &mut Value) -> Option<&mut Vec<Value>> {
    body.get_mut("input").and_then(Value::as_array_mut)
}
#[cfg(test)]
mod trim_test_helpers {
    use super::*;

    /// [`fit_to_cap`] that panics unless the body was trimmed to fit.
    pub trait Trimmed {
        fn trimmed(self) -> TrimReport;
    }

    impl Trimmed for Fitted {
        fn trimmed(self) -> TrimReport {
            match self {
                Fitted::Trimmed(report) => report,
                other => panic!("expected a trim, got {other:?}"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::trim_test_helpers::Trimmed;

    fn output(call_id: &str, text: String) -> Value {
        json!({ "type": "function_call_output", "call_id": call_id, "output": text })
    }

    #[test]
    fn clauses_read_like_english() {
        let report = TrimReport {
            bytes_saved: 1_000,
            removed_images: 1,
            truncated_outputs: 2,
            trimmed_message_parts: 0,
            stripped_reasoning_blobs: 0,
        };
        assert_eq!(report.clause(), "1 image, 2 tool outputs");
        assert_eq!(TrimReport::default().clause(), "nothing");
    }

    #[test]
    fn a_body_that_already_fits_is_left_alone() {
        let mut body = json!({ "input": [output("1", "small".to_string())] });
        let before = body.clone();
        assert!(matches!(fit_to_cap(&mut body, 10_000), Fitted::Untouched));
        assert_eq!(body, before);
    }

    #[test]
    fn the_largest_output_is_cut_until_the_body_fits() {
        let big = "A".repeat(20_000);
        let small = "S".repeat(1_000);
        let mut body = json!({ "input": [output("1", big), output("2", small.clone())] });
        let cap = serialized_len(&body) - 15_000;

        let report = fit_to_cap(&mut body, cap).trimmed();

        assert!(
            serialized_len(&body) <= cap,
            "still {} bytes with a {cap} cap",
            serialized_len(&body)
        );
        assert!(report.bytes_saved >= 15_000, "{report:?}");
        assert_eq!(report.truncated_outputs, 1);

        let trimmed = body["input"][0]["output"].as_str().expect("string");
        assert!(trimmed.starts_with("AAAA"), "the opening is kept");
        assert!(trimmed.ends_with("AAAA"), "the closing is kept");
        assert!(trimmed.contains("codex-router:"), "{trimmed}");
        assert_eq!(body["input"][1]["output"], small, "small outputs survive");
        assert_eq!(
            body["input"][0]["call_id"], "1",
            "item identity is untouched"
        );
    }

    #[test]
    fn opaque_reasoning_blobs_go_before_any_output_is_cut() {
        let mut body = json!({ "input": [
            { "type": "reasoning", "encrypted_content": "Z".repeat(30_000), "summary": [] },
            output("1", "O".repeat(30_000)),
        ] });
        let text = body["input"][1]["output"].clone();
        let cap = serialized_len(&body) - 20_000;

        let report = fit_to_cap(&mut body, cap).trimmed();

        assert_eq!(report.stripped_reasoning_blobs, 1);
        assert_eq!(report.truncated_outputs, 0);
        assert!(body["input"][0].get("encrypted_content").is_none());
        assert_eq!(body["input"][1]["output"], text, "no output had to be cut");
    }

    #[test]
    fn images_are_dropped_when_text_cannot_absorb_the_cut() {
        let mut body = json!({ "input": [{ "type": "message", "role": "user", "content": [
            { "type": "input_image", "image_url": format!("data:image/png;base64,{}", "i".repeat(200_000)) }
        ] }] });
        let cap = serialized_len(&body) - 100_000;

        let report = fit_to_cap(&mut body, cap).trimmed();

        assert_eq!(report.removed_images, 1);
        assert_eq!(body["input"][0]["content"][0]["type"], "input_text");
        let marker = body["input"][0]["content"][0]["text"]
            .as_str()
            .expect("marker");
        assert!(marker.contains("image removed"), "{marker}");
        assert!(serialized_len(&body) <= cap);
    }

    #[test]
    fn only_as_many_images_as_the_overflow_needs_are_dropped() {
        // Ten 100 KB images and a body about 250 KB over the cap: the three
        // oldest images pay for it and the seven newest stay in the conversation.
        let parts: Vec<Value> = (0..10)
            .map(|i| {
                json!({
                    "type": "input_image",
                    "image_url": format!("data:image/png;base64,{}{i}", "i".repeat(100_000)),
                })
            })
            .collect();
        let mut body =
            json!({ "input": [{ "type": "message", "role": "user", "content": parts }] });
        let cap = serialized_len(&body) - 250_000;

        let report = fit_to_cap(&mut body, cap).trimmed();

        assert!(serialized_len(&body) <= cap);
        assert_eq!(report.removed_images, 3, "{report:?}");
        let parts = body["input"][0]["content"].as_array().expect("parts");
        assert_eq!(
            parts[0]["type"], "input_text",
            "the oldest image goes first"
        );
        assert_eq!(parts[9]["type"], "input_image", "the newest image stays");
    }

    #[test]
    fn images_inside_tool_output_parts_are_seen() {
        // Codex puts screenshots in `function_call_output.output` as content
        // parts rather than in a message's `content`; missing that shape was
        // what left the real 53 MB request untrimmable.
        let mut body = json!({ "input": [{
            "type": "function_call_output",
            "call_id": "1",
            "output": [
                { "type": "input_text", "text": "screenshot taken" },
                { "type": "input_image", "image_url": format!("data:image/png;base64,{}", "i".repeat(200_000)) },
            ],
        }] });
        let cap = serialized_len(&body) - 100_000;

        let report = fit_to_cap(&mut body, cap).trimmed();

        assert_eq!(report.removed_images, 1);
        assert!(serialized_len(&body) <= cap);
        let parts = body["input"][0]["output"].as_array().expect("parts");
        assert_eq!(parts.len(), 2, "the part list keeps its shape");
        assert_eq!(parts[0]["type"], "input_text");
        assert_eq!(
            parts[1]["type"], "input_text",
            "the image part became a marker"
        );
    }

    #[test]
    fn history_attachments_are_dropped_before_tool_output_is_cut() {
        let output = "x".repeat(200_000);
        let mut body = json!({ "input": [
            { "type": "message", "role": "user", "content": [
                { "type": "input_image", "image_url": format!("data:image/png;base64,{}", "i".repeat(200_000)) }
            ] },
            { "type": "function_call_output", "call_id": "1", "output": output.clone() },
            { "type": "message", "role": "user", "content": [{ "type": "input_text", "text": "and now?" }] },
        ] });
        let cap = serialized_len(&body) - 100_000;

        let report = fit_to_cap(&mut body, cap).trimmed();

        assert_eq!(report.removed_images, 1, "{report:?}");
        assert_eq!(report.truncated_outputs, 0, "{report:?}");
        assert_eq!(body["input"][0]["content"][0]["type"], "input_text");
        assert_eq!(
            body["input"][1]["output"], output,
            "the tool output stays whole"
        );
    }

    #[test]
    fn the_message_being_answered_keeps_its_attachment() {
        let mut body = json!({ "input": [
            { "type": "message", "role": "user", "content": [{ "type": "input_text", "text": "hi" }] },
            { "type": "function_call_output", "call_id": "1", "output": "y".repeat(400_000) },
            { "type": "message", "role": "user", "content": [
                { "type": "input_image", "image_url": format!("data:image/png;base64,{}", "i".repeat(300_000)) }
            ] },
        ] });
        let cap = serialized_len(&body) - 120_000;

        let report = fit_to_cap(&mut body, cap).trimmed();

        assert_eq!(report.removed_images, 0, "{report:?}");
        assert_eq!(report.truncated_outputs, 1, "{report:?}");
        assert_eq!(body["input"][2]["content"][0]["type"], "input_image");
    }

    #[test]
    fn the_asks_own_attachment_is_the_last_thing_dropped() {
        // Nothing else can pay for the overflow, so the turn is kept and the
        // attachment in it goes — marked, so the model can say what it lost.
        let mut body = json!({ "input": [
            { "type": "function_call_output", "call_id": "1", "output": "y".repeat(100_000) },
            { "type": "message", "role": "user", "content": [
                { "type": "input_image", "image_url": format!("data:image/png;base64,{}", "i".repeat(300_000)) }
            ] },
        ] });
        let cap = serialized_len(&body) - 200_000;

        let report = fit_to_cap(&mut body, cap).trimmed();

        assert_eq!(report.removed_images, 1, "{report:?}");
        assert_eq!(report.truncated_outputs, 1, "the tool output pays first");
        assert_eq!(body["input"][1]["content"][0]["type"], "input_text");
        assert!(serialized_len(&body) <= cap);
    }

    #[test]
    fn pasted_attachments_in_history_are_cut_not_dropped() {
        let pasted = "P".repeat(100_000);
        let mut body = json!({ "input": [
            { "type": "message", "role": "user", "content": [{ "type": "input_text", "text": pasted }] },
            { "type": "message", "role": "user", "content": [{ "type": "input_text", "text": "what changed?" }] },
        ] });
        let cap = serialized_len(&body) - 50_000;

        let report = fit_to_cap(&mut body, cap).trimmed();

        assert_eq!(report.trimmed_message_parts, 1, "{report:?}");
        assert_eq!(report.truncated_outputs, 0, "{report:?}");
        let cut = body["input"][0]["content"][0]["text"]
            .as_str()
            .expect("text");
        assert!(cut.starts_with("PPPP") && cut.contains("codex-router:"));
        assert_eq!(
            body["input"][1]["content"][0]["text"], "what changed?",
            "the question itself is untouched"
        );
    }

    #[test]
    fn short_history_text_is_left_alone() {
        let mut body = json!({ "input": [
            { "type": "message", "role": "user", "content": [{ "type": "input_text", "text": "first request" }] },
            { "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": "did it" }] },
            { "type": "message", "role": "user", "content": [{ "type": "input_text", "text": "and now?" }] },
        ] });
        let before = body.clone();
        assert!(
            matches!(fit_to_cap(&mut body, 40), Fitted::CannotFit(_)),
            "nothing here is an attachment"
        );
        assert_eq!(body, before);
    }

    #[test]
    fn message_and_user_text_is_never_rewritten() {
        let mut body = json!({ "input": [{ "type": "message", "role": "user", "content": [
            { "type": "input_text", "text": "U".repeat(50_000) }
        ] }] });
        let before = body.clone();
        assert!(matches!(fit_to_cap(&mut body, 1_000), Fitted::CannotFit(_)));
        assert_eq!(body, before, "user text is not ours to trim");
    }

    #[test]
    fn refuses_when_even_the_floor_is_too_big() {
        let outputs: Vec<Value> = (0..50)
            .map(|i| output(&i.to_string(), "x".repeat(20_000)))
            .collect();
        let mut body = json!({ "input": outputs });
        let Fitted::CannotFit(report) = fit_to_cap(&mut body, 8_000) else {
            panic!("every output is already at the floor");
        };
        assert!(report.bytes_saved > 0, "it tried: {report:?}");
    }

    #[test]
    fn trimming_keeps_item_order_ids_and_pairing() {
        let mut body = json!({ "input": [
            { "type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}" },
            output("c1", "B".repeat(40_000)),
            { "type": "message", "role": "user", "content": [{ "type": "input_text", "text": "and then?" }] },
            { "type": "function_call", "call_id": "c2", "name": "shell", "arguments": "{}" },
            output("c2", "C".repeat(40_000)),
        ] });
        let cap = serialized_len(&body) - 30_000;

        fit_to_cap(&mut body, cap).trimmed();

        let ids: Vec<&str> = body["input"]
            .as_array()
            .expect("items")
            .iter()
            .filter_map(|item| item.get("call_id").and_then(Value::as_str))
            .collect();
        assert_eq!(
            ids,
            ["c1", "c1", "c2", "c2"],
            "calls keep their outputs, in order"
        );
        assert_eq!(body["input"][2]["content"][0]["text"], "and then?");
    }

    #[test]
    fn cuts_land_on_character_boundaries() {
        // Three-byte characters: half the byte count is mid-character.
        let mut body = json!({ "input": [output("1", "€".repeat(20_000))] });
        let cap = serialized_len(&body) / 2;

        fit_to_cap(&mut body, cap).trimmed();

        let text = body["input"][0]["output"].as_str().expect("string");
        assert!(text.contains("codex-router:"), "{text}");
        assert!(text.starts_with('€') && text.ends_with('€'));
    }

    #[test]
    fn trimming_is_deterministic() {
        let build = || {
            json!({ "input": [
                output("1", "A".repeat(9_000)),
                output("2", "B".repeat(7_000)),
                output("3", "C".repeat(5_000)),
            ] })
        };
        let (mut first, mut second) = (build(), build());
        let cap = serialized_len(&first) - 12_000;

        let left = fit_to_cap(&mut first, cap).trimmed();
        let right = fit_to_cap(&mut second, cap).trimmed();

        assert_eq!(left, right);
        assert_eq!(first, second);
    }

    #[test]
    fn a_body_without_items_is_not_a_crash() {
        let mut body = json!({ "model": "deepseek-flash" });
        assert!(matches!(fit_to_cap(&mut body, 1), Fitted::CannotFit(_)));
    }
}

/// The incident this module exists for: a forked thread replaying a thousand
/// truncated tool outputs lands a few percent over the provider's cap. The trim
/// has to land under the cap while removing *only* that few percent — a rule
/// that gutted the history to be safe would be worse than the 413.
#[cfg(test)]
mod scale_tests {
    use super::*;
    use crate::trim::trim_test_helpers::Trimmed;

    #[test]
    fn a_fork_replay_just_over_the_cap_loses_only_the_overflow() {
        let outputs: Vec<Value> = (0..800)
            .map(|i| {
                json!({
                    "type": "function_call_output",
                    "call_id": format!("call_{i}"),
                    "output": "y".repeat(48_000),
                })
            })
            .collect();
        let mut body = json!({ "input": outputs });
        let cap = serialized_len(&body) - 1_200_000;

        let report = fit_to_cap(&mut body, cap).trimmed();

        assert!(serialized_len(&body) <= cap);
        assert!(
            report.bytes_saved < 2_000_000,
            "cut {} for a 1.2 MB overflow: {:?}",
            report.bytes_saved,
            report
        );
        assert!(report.truncated_outputs > 0);
        // Every output is still present, ids intact, none cut below the floor.
        let items = body["input"].as_array().expect("items");
        assert_eq!(items.len(), 800);
        for (i, item) in items.iter().enumerate() {
            assert_eq!(item["call_id"], format!("call_{i}"));
            assert!(item["output"].as_str().expect("output").len() >= 2 * 1024);
        }
    }
}
