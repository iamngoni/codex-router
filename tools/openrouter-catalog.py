#!/usr/bin/env python3
"""Makes OpenRouter's models selectable in Codex, by generating catalog entries.

Codex resolves model names through `model_catalog_json`
(`~/.codex/model-catalogs/all.json`): an entry's `slug` is the string Codex
sends as `model`, and that string has to be `openrouter/<vendor>/<model>` so
codex-router's `openrouter/` route matches and strips its own prefix before
forwarding. Nothing else in the catalog supplies those names, so this script
fetches OpenRouter's public model list and writes the entries.

Entries mirror the shape of the existing third-party entries (DeepSeek, Z.ai):
`context_window` comes from OpenRouter's `context_length`, `input_modalities`
is narrowed to what Codex understands (text/image), and reasoning levels are
only offered for models whose `supported_parameters` say they take reasoning —
otherwise Codex would send an effort the model cannot honour.

By default it considers only models Codex can actually drive: tool calling,
text in and out, no `:variant` suffixes, no dated snapshots. Pass `--all` for
the raw list.

    python3 tools/openrouter-catalog.py              # preview: summary + sample
    python3 tools/openrouter-catalog.py --install    # merge into all.json (backed up)
    python3 tools/openrouter-catalog.py --keep-variants --install   # + :free/:nitro variants
    python3 tools/openrouter-catalog.py --all --install             # everything the API lists

Existing entries are preserved; only `openrouter/` entries are replaced, so
re-running after OpenRouter ships new models is safe.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import urllib.request

MODELS_URL = "https://openrouter.ai/api/v1/models"
CATALOG = os.path.expanduser("~/.codex/model-catalogs/all.json")
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
    return catalog, len(existing) - len(kept)


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
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(catalog, handle, indent=1)
        print(f"\nwrote {args.out}")
    elif args.install:
        if os.path.exists(CATALOG):
            backup = f"{CATALOG}.{time.strftime('%Y%m%d-%H%M%S')}.bak"
            shutil.copy2(CATALOG, backup)
            print(f"\nbackup: {backup}")
        os.makedirs(os.path.dirname(CATALOG), exist_ok=True)
        with open(CATALOG, "w", encoding="utf-8") as handle:
            json.dump(catalog, handle, indent=1)
        print(f"wrote {CATALOG} — restart Codex to pick the new models up")
    else:
        print("\n(preview only — pass --install to write the catalog)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
