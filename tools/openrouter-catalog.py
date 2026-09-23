#!/usr/bin/env python3
"""Makes OpenRouter's models selectable in Codex, by generating catalog entries.

The router merges entries from
`~/.codex/model-catalogs/external.json` into Codex's native model catalogue.
An entry's `slug` is the string Codex sends as `model`, and that string has to
be `openrouter/<vendor>/<model>` so codex-router's `openrouter/` route matches
and strips its own prefix before forwarding. This script fetches OpenRouter's
public model list and writes only those external entries.

Entries mirror the shape of the existing third-party entries (DeepSeek, Z.ai):
`context_window` comes from OpenRouter's `context_length`, `input_modalities`
is narrowed to what Codex understands (text/image), and reasoning levels are
only offered for models whose `supported_parameters` say they take reasoning —
otherwise Codex would send an effort the model cannot honour.

By default it considers only models Codex can actually drive: tool calling,
text in and out, no `:variant` suffixes, no dated snapshots. Pass `--all` for
the raw list.

    python3 tools/openrouter-catalog.py              # preview: summary + sample
    python3 tools/openrouter-catalog.py --install    # merge into external.json (backed up)
    python3 tools/openrouter-catalog.py --keep-variants --install   # + :free/:nitro variants
    python3 tools/openrouter-catalog.py --all --install             # everything the API lists

Existing entries are preserved; only `openrouter/` entries are replaced, so
manual DeepSeek/GLM entries in the external catalogue survive a refresh.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
import urllib.request

MODELS_URL = "https://openrouter.ai/api/v1/models"
CATALOG = os.path.expanduser("~/.codex/model-catalogs/external.json")
SLUG_PREFIX = "openrouter/"

# Copied from the existing third-party entries so OpenRouter models behave like
# DeepSeek's and Z.ai's rather than inventing a new shape.
BASE_ENTRY = {
    "shell_type": "shell_command",
    "visibility": "list",
    "supported_in_api": True,
    "priority": 20,
    "base_instructions": (
        "You are a coding assistant. Help the user complete their task accurately, "
        "use available tools, and verify your changes."
    ),
    "effective_context_window_percent": 95,
    "truncation_policy": {"mode": "tokens", "limit": 10000},
    "apply_patch_tool_type": "freeform",
    "support_verbosity": True,
    "default_verbosity": "low",
    "default_reasoning_summary": "none",
    "supports_parallel_tool_calls": True,
    "use_responses_lite": False,
    "prefer_websockets": False,
    "experimental_supported_tools": [],
}

REASONING_LEVELS = [
    {"effort": "low", "description": "Low reasoning"},
    {"effort": "high", "description": "High reasoning"},
    {"effort": "max", "description": "Max reasoning"},
]

DATED = re.compile(r"-(?:\d{8}|\d{4}-\d{2}-\d{2}|v\d+)$")


def fetch_models() -> list[dict]:
    with urllib.request.urlopen(MODELS_URL, timeout=30) as response:
        return json.load(response)["data"]


def usable(model: dict) -> bool:
    """Whether Codex can drive this model at all."""
    arch = model.get("architecture") or {}
    params = model.get("supported_parameters") or []
    return (
        "tools" in params
        and "text" in (arch.get("input_modalities") or [])
        and "text" in (arch.get("output_modalities") or [])
    )


def canonical(model: dict) -> bool:
    """Drops `:free`/`:nitro` style variants and pinned dated snapshots, which
    would otherwise fill the picker with near-duplicates."""
    model_id = model["id"]
    return (
        not model_id.startswith("~")
        and ":" not in model_id
        and not DATED.search(model_id)
    )


def entry_for(model: dict) -> dict:
    arch = model.get("architecture") or {}
    modalities = ["text"]
    if "image" in (arch.get("input_modalities") or []):
        modalities.append("image")
    context = model.get("context_length") or 128_000

    entry = dict(BASE_ENTRY)
    entry.update(
        {
            "slug": SLUG_PREFIX + model["id"],
            "display_name": f"{model['name']} (OpenRouter)",
            "description": f"{model['name']} via your OpenRouter account.",
            "context_window": context,
            "max_context_window": context,
            "input_modalities": modalities,
        }
    )
    if "reasoning" in (model.get("supported_parameters") or []):
        entry["default_reasoning_level"] = "high"
        entry["supported_reasoning_levels"] = REASONING_LEVELS
    else:
        entry["supported_reasoning_levels"] = []
    return entry


def merge(catalog_path: str, entries: list[dict]) -> tuple[dict, int]:
    catalog = {"models": []}
    if os.path.exists(catalog_path):
        with open(catalog_path, encoding="utf-8") as handle:
            catalog = json.load(handle)
    existing = catalog.get("models", [])
    kept = [m for m in existing if not str(m.get("slug", "")).startswith(SLUG_PREFIX)]
    catalog["models"] = kept + entries
    validate_catalog(catalog)
    return catalog, len(existing) - len(kept)


def validate_catalog(catalog: dict) -> None:
    """Reject malformed output before it can replace the installed file."""
    if not isinstance(catalog, dict) or not isinstance(catalog.get("models"), list):
        raise ValueError("catalogue must be an object with a models list")
    seen: set[str] = set()
    for model in catalog["models"]:
        if not isinstance(model, dict):
            raise ValueError("catalogue models must be objects")
        slug = model.get("slug")
        if not isinstance(slug, str) or not slug.strip() or slug.strip() in seen:
            raise ValueError("catalogue slugs must be nonempty and unique")
        seen.add(slug.strip())
        _validate_model_fields(model)


def _validate_model_fields(model: dict) -> None:
    string_fields = {
        "slug",
        "display_name",
        "base_instructions",
        "shell_type",
        "visibility",
        "default_reasoning_summary",
        "comp_hash",
        "model_specialty",
        "multi_agent_reasoning_effort",
        "multi_agent_version",
        "tool_mode",
        "web_search_tool_type",
    }
    optional_string_fields = {
        "description",
        "default_reasoning_level",
        "default_verbosity",
        "apply_patch_tool_type",
    }
    nullable_integer_fields = {
        "context_window",
        "max_context_window",
        "auto_compact_token_limit",
    }
    boolean_fields = {
        "supported_in_api",
        "support_verbosity",
        "supports_parallel_tool_calls",
        "use_responses_lite",
        "prefer_websockets",
        "supports_search_tool",
        "supports_image_detail_original",
        "supports_experimental_context",
        "supports_reasoning_summary_parameter",
        "include_apps_usage_instructions",
        "include_plugin_usage_instructions",
        "include_skills_usage_instructions",
        "node_repl_disabled",
        "node_repl_auto_review_required",
    }
    for key in string_fields:
        if key in model and not isinstance(model[key], str):
            raise ValueError(f"catalogue field {key} must be a string")
    for key in optional_string_fields:
        if key in model and model[key] is not None and not isinstance(model[key], str):
            raise ValueError(f"catalogue field {key} must be a string or null")
    for key in nullable_integer_fields:
        if key in model and model[key] is not None and not _is_i64(model[key]):
            raise ValueError(f"catalogue field {key} must be an integer or null")
    for key in ("priority", "effective_context_window_percent"):
        if key in model and not _is_i64(model[key]):
            raise ValueError(f"catalogue field {key} must be an integer")
    if "priority" in model and not -(1 << 31) <= model["priority"] < (1 << 31):
        raise ValueError("catalogue field priority must fit in a signed 32-bit integer")
    for key in boolean_fields:
        if key in model and not isinstance(model[key], bool):
            raise ValueError(f"catalogue field {key} must be a boolean")

    string_arrays = (
        "additional_speed_tiers",
        "experimental_supported_tools",
        "input_modalities",
    )
    for key in string_arrays:
        if key in model and not _is_array_of(model[key], str):
            raise ValueError(f"catalogue field {key} must be an array of strings")
    if "service_tiers" in model:
        tiers = model["service_tiers"]
        if not isinstance(tiers, list) or any(
            not isinstance(tier, dict)
            or any(not isinstance(tier.get(key), str) for key in ("id", "name", "description"))
            for tier in tiers
        ):
            raise ValueError(
                "catalogue field service_tiers must contain objects with string id, name, and description"
            )
    if "supported_reasoning_levels" in model:
        levels = model["supported_reasoning_levels"]
        if not isinstance(levels, list) or any(
            not isinstance(level, dict)
            or not isinstance(level.get("effort"), str)
            or not isinstance(level.get("description"), str)
            for level in levels
        ):
            raise ValueError(
                "catalogue field supported_reasoning_levels must contain objects with string effort and description"
            )

    if "truncation_policy" in model and not _valid_truncation_policy(model["truncation_policy"]):
        raise ValueError("catalogue field truncation_policy must contain a string mode and integer limit")
    if "availability_nux" in model and model["availability_nux"] is not None:
        availability = model["availability_nux"]
        if not isinstance(availability, dict) or not isinstance(availability.get("message"), str):
            raise ValueError("catalogue field availability_nux must be an object with a string message or null")
    if "upgrade" in model and model["upgrade"] is not None and not _valid_upgrade(model["upgrade"]):
        raise ValueError(
            "catalogue field upgrade must be an object with string model and migration_markdown fields or null"
        )
    if "model_messages" in model and model["model_messages"] is not None:
        messages = model["model_messages"]
        if not isinstance(messages, dict):
            raise ValueError("catalogue field model_messages must be an object or null")
        template = messages.get("instructions_template")
        if template is not None and not isinstance(template, str):
            raise ValueError("catalogue field model_messages.instructions_template must be a string or null")
        variables = messages.get("instructions_variables")
        if variables is not None:
            variable_names = (
                "personality_default",
                "personality_friendly",
                "personality_pragmatic",
            )
            if not isinstance(variables, dict) or any(
                name in variables
                and variables[name] is not None
                and not isinstance(variables[name], str)
                for name in variable_names
            ):
                raise ValueError("catalogue field model_messages.instructions_variables is malformed")


def _is_i64(value: object) -> bool:
    return type(value) is int and -(1 << 63) <= value < (1 << 63)


def _is_array_of(value: object, item_type: type) -> bool:
    return isinstance(value, list) and all(isinstance(item, item_type) for item in value)


def _valid_truncation_policy(value: object) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("mode"), str)
        and _is_i64(value.get("limit"))
    )


def _valid_upgrade(value: object) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("model"), str)
        and isinstance(value.get("migration_markdown"), str)
    )


def write_atomic(path: str, catalog: dict) -> None:
    """Validate and replace a catalogue without exposing a partial write."""
    validate_catalog(catalog)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".external.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(catalog, handle, indent=1)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help=f"write into {CATALOG} (backed up first)")
    parser.add_argument("--all", action="store_true", help="keep every model, not just ones Codex can drive")
    parser.add_argument("--keep-variants", action="store_true", help="also keep :free/:nitro variants and dated snapshots")
    parser.add_argument("--from-file", help="read the model list from a saved JSON file instead of the API")
    parser.add_argument("--out", help="write the merged catalog here instead of installing")
    args = parser.parse_args()

    if args.from_file:
        with open(args.from_file, encoding="utf-8") as handle:
            models = json.load(handle)["data"]
    else:
        models = fetch_models()

    chosen = [
        m
        for m in models
        if (args.all or usable(m)) and (args.all or args.keep_variants or canonical(m))
    ]
    entries = [entry_for(m) for m in sorted(chosen, key=lambda m: m["id"])]
    if args.all:
        how = "unfiltered"
    elif args.keep_variants:
        how = "tool-capable, variants included"
    else:
        how = "tool-capable, canonical ids only"
    print(f"{len(models)} models at OpenRouter -> {len(entries)} catalog entries ({how})")

    target = args.out or CATALOG
    catalog, replaced = merge(target, entries)
    print(
        f"catalog would hold {len(catalog['models'])} models "
        f"({replaced} existing openrouter/ entries replaced, everything else kept)"
    )
    if entries:
        print("\nsample entry:\n" + json.dumps(entries[0], indent=1))

    if args.out:
        write_atomic(args.out, catalog)
        print(f"\nwrote {args.out}")
    elif args.install:
        if os.path.exists(CATALOG):
            backup = f"{CATALOG}.{time.strftime('%Y%m%d-%H%M%S')}.bak"
            with open(CATALOG, encoding="utf-8") as source, open(backup, "w", encoding="utf-8") as target:
                target.write(source.read())
            print(f"\nbackup: {backup}")
        write_atomic(CATALOG, catalog)
        print(f"wrote {CATALOG} — restart Codex to pick the new models up")
    else:
        print("\n(preview only — pass --install to write the catalog)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
