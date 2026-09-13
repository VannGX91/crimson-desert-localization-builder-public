import os
import tempfile
import unittest
from pathlib import Path

from crimson0020.translation_runtime.knowledge.database import (
    KnowledgeDatabase,
    resolve_default_database_path,
)
from crimson0020.translation_runtime.knowledge.schema import SCHEMA_VERSION


class TestTranslationKnowledgeDatabase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "test_knowledge.sqlite3"

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_create_and_reopen(self):
        db = KnowledgeDatabase(self.db_path)
        with db.connect() as con:
            cur = con.cursor()
            cur.execute("SELECT schema_version FROM schema_metadata")
            row = cur.fetchone()
            self.assertEqual(row[0], SCHEMA_VERSION)

        # Reopen
        db2 = KnowledgeDatabase(self.db_path)
        with db2.connect() as con:
            cur = con.cursor()
            cur.execute("SELECT schema_version FROM schema_metadata")
            row = cur.fetchone()
            self.assertEqual(row[0], SCHEMA_VERSION)

    def test_wal_and_busy_timeout(self):
        db = KnowledgeDatabase(self.db_path)
        with db.connect() as con:
            cur = con.cursor()
            cur.execute("PRAGMA journal_mode")
            mode = cur.fetchone()[0]
            self.assertEqual(mode.lower(), "wal")

    def test_transaction_rollback(self):
        db = KnowledgeDatabase(self.db_path)
        try:
            with db.transaction() as con:
                con.execute(
                    "INSERT INTO scopes (scope_id, app_id, package_family, source_language, target_language, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    ("sc1", "app1", "pkg1", "en", "vi", 1000.0)
                )
                raise RuntimeError("Simulated crash")
        except RuntimeError:
            pass

        with db.connect() as con:
            cur = con.cursor()
            cur.execute("SELECT COUNT(*) FROM scopes")
            self.assertEqual(cur.fetchone()[0], 0)

    def test_fts5_virtual_tables(self):
        db = KnowledgeDatabase(self.db_path)
        with db.transaction() as con:
            con.execute(
                "INSERT INTO scopes (scope_id, app_id, package_family, source_language, target_language, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                ("sc1", "app1", "pkg1", "en", "vi", 1000.0)
            )
            con.execute(
                "INSERT INTO translation_segments (segment_id, scope_id, lineage_id, marker, key, source_hash, source_text, target_text, context_signature, hard_terms_hash, glossary_revision, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("seg1", "sc1", "lin1", "m", "k", "h1", "The golden sword of dawn", "Thanh gươm hoàng kim của bình minh", "ctx1", "", 1, "active", 1000.0, 1000.0)
            )

        with db.connect() as con:
            cur = con.cursor()
            cur.execute("SELECT segment_id FROM segments_fts WHERE segments_fts MATCH 'golden sword'")
            rows = cur.fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][0], "seg1")

    def test_integrity_check_and_corruption_recovery(self):
        db = KnowledgeDatabase(self.db_path)
        with db.transaction() as con:
            con.execute(
                "INSERT INTO scopes (scope_id, app_id, package_family, source_language, target_language, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                ("sc1", "app1", "pkg1", "en", "vi", 1000.0)
            )

        # Corrupt the database file
        with open(self.db_path, "r+b") as f:
            f.seek(16)
            f.write(b"\xff" * 64)

        # Reopening should trigger recovery
        db_recovered = KnowledgeDatabase(self.db_path)
        recovery_result = db_recovered.last_recovery_result
        self.assertTrue(recovery_result.recovered)
        self.assertIsNotNone(recovery_result.backup_path)
        self.assertTrue(Path(recovery_result.backup_path).exists())

        # Verify new clean database is functional
        with db_recovered.connect() as con:
            cur = con.cursor()
            cur.execute("SELECT COUNT(*) FROM scopes")
            self.assertEqual(cur.fetchone()[0], 0)

    def test_scope_isolation(self):
        db = KnowledgeDatabase(self.db_path)
        with db.transaction() as con:
            con.execute(
                "INSERT INTO scopes (scope_id, app_id, package_family, source_language, target_language, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                ("sc_en_vi", "game1", "core", "en", "vi", 1000.0)
            )
            con.execute(
                "INSERT INTO scopes (scope_id, app_id, package_family, source_language, target_language, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                ("sc_en_fr", "game1", "core", "en", "fr", 1000.0)
            )
            con.execute(
                "INSERT INTO translation_segments (segment_id, scope_id, lineage_id, marker, key, source_hash, source_text, target_text, context_signature, hard_terms_hash, glossary_revision, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("seg_vi", "sc_en_vi", "lin1", "m", "k", "h", "Hello", "Xin chào", "ctx", "", 1, "active", 1000.0, 1000.0)
            )
            con.execute(
                "INSERT INTO translation_segments (segment_id, scope_id, lineage_id, marker, key, source_hash, source_text, target_text, context_signature, hard_terms_hash, glossary_revision, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("seg_fr", "sc_en_fr", "lin1", "m", "k", "h", "Hello", "Bonjour", "ctx", "", 1, "active", 1000.0, 1000.0)
            )

        with db.connect() as con:
            cur = con.cursor()
            cur.execute("SELECT target_text FROM translation_segments WHERE scope_id = 'sc_en_vi'")
            self.assertEqual(cur.fetchone()[0], "Xin chào")
            cur.execute("SELECT target_text FROM translation_segments WHERE scope_id = 'sc_en_fr'")
            self.assertEqual(cur.fetchone()[0], "Bonjour")

    def test_default_path_resolution(self):
        old_env = os.environ.get("LOCALAPPDATA")
        try:
            os.environ["LOCALAPPDATA"] = str(Path(self.tmp_dir.name) / "localapp")
            p = resolve_default_database_path()
            self.assertEqual(p.name, "translation-knowledge-v1.sqlite3")
            self.assertTrue(p.parent.exists())

            # Fallback when LOCALAPPDATA is absent
            del os.environ["LOCALAPPDATA"]
            p2 = resolve_default_database_path()
            self.assertEqual(p2.name, "translation-knowledge-v1.sqlite3")
        finally:
            if old_env is not None:
                os.environ["LOCALAPPDATA"] = old_env

    def test_concurrent_readers(self):
        db = KnowledgeDatabase(self.db_path)
        with db.connect() as con1:
            with db.connect() as con2:
                cur1 = con1.cursor()
                cur2 = con2.cursor()
                cur1.execute("SELECT COUNT(*) FROM scopes")
                cur2.execute("SELECT COUNT(*) FROM scopes")
                self.assertEqual(cur1.fetchone()[0], 0)
                self.assertEqual(cur2.fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
