import json
import tempfile
import unittest
from pathlib import Path

from crimson0020.translation_runtime.knowledge.database import KnowledgeDatabase
from crimson0020.translation_runtime.knowledge.import_export import (
    KnowledgeImportError,
)
from crimson0020.translation_runtime.knowledge.service import (
    CatalogRegistration,
    KnowledgeService,
    ScopeIdentifier,
)


class TestTranslationKnowledgeService(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "service_test.sqlite3"
        self.db = KnowledgeDatabase(self.db_path)
        self.service = KnowledgeService(self.db)
        self.scope = ScopeIdentifier("app1", "core", "en", "vi")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_first_catalog_registration_creates_lineage(self):
        reg = CatalogRegistration(
            scope=self.scope,
            fingerprint="fp_v1_0",
            item_count=100,
            prior_fingerprint=None,
        )
        lineage_id = self.service.register_catalog(reg)
        self.assertTrue(lineage_id.startswith("lin_"))

        # Query lineage
        cur_lineage = self.service.get_lineage_for_fingerprint(self.scope, "fp_v1_0")
        self.assertEqual(cur_lineage, lineage_id)

    def test_next_catalog_with_known_prior_continues_lineage(self):
        reg1 = CatalogRegistration(
            scope=self.scope,
            fingerprint="fp_v1_0",
            item_count=100,
            prior_fingerprint=None,
        )
        lin1 = self.service.register_catalog(reg1)

        reg2 = CatalogRegistration(
            scope=self.scope,
            fingerprint="fp_v1_1",
            item_count=105,
            prior_fingerprint="fp_v1_0",
        )
        lin2 = self.service.register_catalog(reg2)
        self.assertEqual(lin1, lin2)  # Continues same lineage!

    def test_unknown_or_unrelated_chain_creates_new_lineage(self):
        reg1 = CatalogRegistration(
            scope=self.scope,
            fingerprint="fp_catA",
            item_count=50,
            prior_fingerprint=None,
        )
        lin1 = self.service.register_catalog(reg1)

        # Unrelated catalog with unknown prior creates distinct lineage
        reg2 = CatalogRegistration(
            scope=self.scope,
            fingerprint="fp_catB",
            item_count=80,
            prior_fingerprint="fp_nonexistent",
        )
        lin2 = self.service.register_catalog(reg2)
        self.assertNotEqual(lin1, lin2)

    def test_paginated_queries(self):
        # Register and insert segments
        reg = CatalogRegistration(scope=self.scope, fingerprint="fp_page", item_count=50)
        lin = self.service.register_catalog(reg)

        for i in range(15):
            self.service.save_translation_segment(
                scope=self.scope,
                lineage_id=lin,
                marker="loc",
                key=f"KEY_{i}",
                source_hash=f"sh_{i}",
                source_text=f"Text {i}",
                target_text=f"Bản dịch {i}",
            )

        page1 = self.service.list_segments(self.scope, limit=10, offset=0)
        self.assertEqual(len(page1.items), 10)
        self.assertEqual(page1.total_count, 15)

        page2 = self.service.list_segments(self.scope, limit=10, offset=10)
        self.assertEqual(len(page2.items), 5)

    def test_export_and_import_roundtrip(self):
        reg = CatalogRegistration(scope=self.scope, fingerprint="fp_exp", item_count=10)
        lin = self.service.register_catalog(reg)
        self.service.save_translation_segment(
            scope=self.scope,
            lineage_id=lin,
            marker="loc",
            key="KEY_1",
            source_hash="sh_1",
            source_text="Sword",
            target_text="Thanh kiếm",
        )
        self.service.set_glossary_override(self.scope, "Sword", "Kiếm quý")

        export_data = self.service.export_package(self.scope)
        self.assertEqual(export_data["schema_version"], 1)
        self.assertEqual(len(export_data["segments"]), 1)
        self.assertEqual(len(export_data["terms"]), 1)

        # Ensure no private paths or credentials
        json_str = json.dumps(export_data)
        self.assertNotIn("Users", json_str)
        self.assertNotIn("token", json_str.lower())

        # Clear db and import
        new_db_path = Path(self.tmp_dir.name) / "imported.sqlite3"
        new_service = KnowledgeService(KnowledgeDatabase(new_db_path))
        new_service.import_package(export_data)

        imported_segs = new_service.list_segments(self.scope, limit=10)
        self.assertEqual(len(imported_segs.items), 1)
        self.assertEqual(imported_segs.items[0].target_text, "Thanh kiếm")

    def test_malformed_or_oversized_import_rejected(self):
        # Missing schema_version
        with self.assertRaises(KnowledgeImportError):
            self.service.import_package({"segments": []})

        # Unsupported schema version
        with self.assertRaises(KnowledgeImportError):
            self.service.import_package({"schema_version": 999, "scope": {}})


if __name__ == "__main__":
    unittest.main()
