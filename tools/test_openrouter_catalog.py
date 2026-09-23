import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("openrouter-catalog.py")
SPEC = importlib.util.spec_from_file_location("openrouter_catalog", MODULE_PATH)
catalog = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(catalog)


class CatalogTests(unittest.TestCase):
    def test_merge_replaces_only_openrouter_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external.json"
            path.write_text(
                json.dumps(
                    {
                        "models": [
                            {"slug": "deepseek-chat", "priority": 1},
                            {"slug": "openrouter/old", "priority": 20},
                        ],
                        "metadata": {"keep": True},
                    }
                ),
                encoding="utf-8",
            )
            merged, replaced = catalog.merge(
                str(path), [{"slug": "openrouter/new", "priority": 20}]
            )
            self.assertEqual(replaced, 1)
            self.assertEqual(
                [model["slug"] for model in merged["models"]],
                ["deepseek-chat", "openrouter/new"],
            )
            self.assertEqual(merged["metadata"], {"keep": True})

    def test_write_atomic_validates_and_replaces(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external.json"
            catalog.write_atomic(str(path), {"models": [{"slug": "openrouter/new"}]})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["models"][0]["slug"], "openrouter/new")
            original = path.read_text(encoding="utf-8")
            invalid_upgrades = [
                {"model": 7, "migration_markdown": "Move over."},
                {"model": "next-model", "migration_markdown": False},
                {"migration_markdown": "Move over."},
                {"model": "next-model"},
            ]
            for upgrade in invalid_upgrades:
                with self.subTest(upgrade=upgrade), self.assertRaises(ValueError):
                    catalog.write_atomic(
                        str(path), {"models": [{"slug": "openrouter/new", "upgrade": upgrade}]}
                    )
                self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_validation_rejects_invalid_known_field_types(self):
        invalid_models = [
            {"slug": "bad-context", "context_window": "large"},
            {"slug": "bad-priority", "priority": True},
            {"slug": "bad-reasoning-number", "supported_reasoning_levels": [1]},
            {
                "slug": "bad-reasoning-shape",
                "supported_reasoning_levels": [{"effort": "high"}],
            },
            {"slug": "bad-modality", "input_modalities": ["text", 1]},
            {"slug": "bad-service-tier", "service_tiers": [False]},
            {"slug": "bad-service-tier-shape", "service_tiers": [{"id": "priority"}]},
            {"slug": "bad-speed-tier", "additional_speed_tiers": [2]},
            {"slug": "bad-tool-list", "experimental_supported_tools": [{}]},
            {"slug": "bad-availability", "availability_nux": {"message": 1}},
            {"slug": "bad-truncation", "truncation_policy": {"mode": "tokens", "limit": "10"}},
            {
                "slug": "bad-model-messages",
                "model_messages": {"instructions_template": 1},
            },
            {"slug": "bad-upgrade-type", "upgrade": "next-model"},
            {"slug": "bad-upgrade-model-type", "upgrade": {"model": 7, "migration_markdown": "Move over."}},
            {"slug": "bad-upgrade-markdown-type", "upgrade": {"model": "next-model", "migration_markdown": False}},
            {"slug": "bad-upgrade-missing-model", "upgrade": {"migration_markdown": "Move over."}},
            {"slug": "bad-upgrade-missing-markdown", "upgrade": {"model": "next-model"}},
        ]
        for model in invalid_models:
            with self.subTest(model=model), self.assertRaises(ValueError):
                catalog.validate_catalog({"models": [model]})

    def test_validation_accepts_nullable_fields_and_object_upgrade(self):
        catalog.validate_catalog(
            {
                "models": [
                    {
                        "slug": "nullable-fields",
                        "description": None,
                        "default_reasoning_level": None,
                        "default_verbosity": None,
                        "apply_patch_tool_type": None,
                        "context_window": None,
                        "max_context_window": None,
                        "auto_compact_token_limit": None,
                        "availability_nux": None,
                        "model_messages": None,
                        "upgrade": {
                            "model": "next-model",
                            "migration_markdown": "Move over.",
                            "future_field": "preserved",
                        },
                        "service_tiers": [
                            {"id": "priority", "name": "Fast", "description": "2x speed"}
                        ],
                        "vendor_extension": {"kept": True},
                    }
                ]
            }
        )


if __name__ == "__main__":
    unittest.main()
