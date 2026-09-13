//! JSON Schema hygiene for tool definitions sent to third-party providers.
//!
//! Owns two independent fixups applied to `tools` before a routed request
//! leaves the process: dropping tool types with no third-party equivalent,
//! and breaking `$ref` cycles that OpenAI tolerates but third-party
//! Responses/Chat-Completions clones reject outright.

use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};

/// Some third-party providers reject tool schemas whose $defs contain a
/// reference cycle (e.g. a MIME-part tree that nests a type inside itself
/// via a `parts` field). OpenAI's own API has no such restriction, so this
/// only runs for routed (non-OpenAI) traffic.
fn collect_ref_names(node: &Value, out: &mut HashSet<String>) {
    match node {
        Value::Array(arr) => arr.iter().for_each(|v| collect_ref_names(v, out)),
        Value::Object(map) => {
            if let Some(Value::String(r)) = map.get("$ref")
                && let Some(name) = r.rsplit('/').next()
            {
                out.insert(name.to_string());
            }
            map.values().for_each(|v| collect_ref_names(v, out));
        }
        _ => {}
    }
}

fn is_cyclic(start: &str, graph: &HashMap<String, HashSet<String>>) -> bool {
    let mut seen: HashSet<String> = HashSet::new();
    let mut stack: Vec<String> = graph
        .get(start)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .collect();
    while let Some(next) = stack.pop() {
        if next == start {
            return true;
        }
        if !seen.insert(next.clone()) {
            continue;
        }
        if let Some(refs) = graph.get(&next) {
            stack.extend(refs.iter().cloned());
        }
    }
    false
}

fn cut_refs_into(node: &mut Value, cyclic: &HashSet<String>) {
    match node {
        Value::Array(arr) => arr.iter_mut().for_each(|v| cut_refs_into(v, cyclic)),
        Value::Object(map) => {
            let matched = matches!(
                map.get("$ref"),
                Some(Value::String(r)) if cyclic.contains(r.rsplit('/').next().unwrap_or(""))
            );
            if matched {
                map.remove("$ref");
                map.insert("type".to_string(), Value::String("object".to_string()));
                map.insert("additionalProperties".to_string(), Value::Bool(true));
                return;
            }
            for v in map.values_mut() {
                cut_refs_into(v, cyclic);
            }
        }
        _ => {}
    }
}

/// $defs live wherever a schema container appears — a plain tool's own
/// `parameters`, or (for MCP namespace tools) each sub-tool's `parameters`.
/// Walk the whole tree and fix every container found, rather than assuming
/// a fixed shape.
fn break_cycles_in_container(container: &mut Map<String, Value>) {
    for key in ["$defs", "definitions"] {
        let Some(Value::Object(mut defs)) = container.remove(key) else {
            continue;
        };

        let mut graph: HashMap<String, HashSet<String>> = HashMap::new();
        for (name, def) in defs.iter() {
            let mut refs = HashSet::new();
            collect_ref_names(def, &mut refs);
            graph.insert(name.clone(), refs);
        }

        let cyclic: HashSet<String> = graph
            .keys()
            .filter(|k| is_cyclic(k, &graph))
            .cloned()
            .collect();
        if !cyclic.is_empty() {
            for name in &cyclic {
                if let Some(def) = defs.get_mut(name) {
                    cut_refs_into(def, &cyclic);
                }
            }
        }
        container.insert(key.to_string(), Value::Object(defs));
    }
}

/// Recursively breaks `$defs`/`definitions` reference cycles anywhere in
/// `node`. Safe to call on any JSON value, not just a tool's `parameters`.
pub fn sanitize_schema_tree(node: &mut Value) {
    match node {
        Value::Array(arr) => arr.iter_mut().for_each(sanitize_schema_tree),
        Value::Object(map) => {
            if map.contains_key("$defs") || map.contains_key("definitions") {
                break_cycles_in_container(map);
            }
            for v in map.values_mut() {
                sanitize_schema_tree(v);
            }
        }
        _ => {}
    }
}

/// OpenAI-only tool types with no equivalent on third-party Responses-API
/// clones: `custom` (freeform grammar tools like apply_patch) and
/// `web_search` (OpenAI-hosted search grounding). Drop them for routed
/// traffic rather than sending a shape the upstream will 400 on.
const UNROUTABLE_TOOL_TYPES: [&str; 2] = ["custom", "web_search"];

/// Drops OpenAI-only tool types and cycle-breaks the rest, in place, on a
/// parsed request body's top-level `tools` array (a no-op if absent).
pub fn sanitize_tools_for_route(parsed: &mut Value) {
    let Some(Value::Array(tools)) = parsed.get_mut("tools") else {
        return;
    };
    tools.retain(|tool| {
        let t = tool.get("type").and_then(Value::as_str).unwrap_or("");
        !UNROUTABLE_TOOL_TYPES.contains(&t)
    });
    for tool in tools.iter_mut() {
        sanitize_schema_tree(tool);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn drops_unroutable_tool_types() {
        let mut body = json!({
            "tools": [
                { "type": "function", "name": "get_weather" },
                { "type": "custom", "name": "apply_patch" },
                { "type": "web_search" },
            ]
        });
        sanitize_tools_for_route(&mut body);
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "get_weather");
    }

    #[test]
    fn breaks_self_referential_cycle() {
        let mut schema = json!({
            "type": "object",
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": { "children": { "type": "array", "items": { "$ref": "#/$defs/Node" } } }
                }
            }
        });
        sanitize_schema_tree(&mut schema);
        let item_ref = &schema["$defs"]["Node"]["properties"]["children"]["items"];
        assert!(item_ref.get("$ref").is_none());
        assert_eq!(item_ref["type"], "object");
        assert_eq!(item_ref["additionalProperties"], true);
    }

    #[test]
    fn leaves_acyclic_refs_untouched() {
        let mut schema = json!({
            "$defs": { "Leaf": { "type": "string" } },
            "properties": { "name": { "$ref": "#/$defs/Leaf" } }
        });
        sanitize_schema_tree(&mut schema);
        assert_eq!(schema["properties"]["name"]["$ref"], "#/$defs/Leaf");
    }
}
