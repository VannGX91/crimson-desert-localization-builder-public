import unittest

from crimson0020.translation_runtime.knowledge.context import (
    ContextEngine,
    ContextKind,
    ContextItem,
)


class TestTranslationContext(unittest.TestCase):
    def setUp(self):
        self.engine = ContextEngine()

    def test_scene_window_plus_minus_two(self):
        # 5 lines in a scene
        lines = [
            ContextItem(index=0, key="L0", text="Line 0", scene_id="scene_intro"),
            ContextItem(index=1, key="L1", text="Line 1", scene_id="scene_intro"),
            ContextItem(index=2, key="L2", text="Line 2", scene_id="scene_intro"),
            ContextItem(index=3, key="L3", text="Line 3", scene_id="scene_intro"),
            ContextItem(index=4, key="L4", text="Line 4", scene_id="scene_intro"),
        ]
        # Target is Line 2: window should include Line 0, 1, 3, 4
        ctx = self.engine.build_context(lines, target_index=2, kind=ContextKind.SCENE)
        self.assertIn("Line 0", ctx.context_text)
        self.assertIn("Line 1", ctx.context_text)
        self.assertIn("Line 3", ctx.context_text)
        self.assertIn("Line 4", ctx.context_text)
        self.assertNotIn("Line 2", ctx.surrounding_text)  # surrounding does not duplicate target
        self.assertTrue(ctx.signature)

    def test_document_window_plus_minus_one_with_title(self):
        doc = [
            ContextItem(index=0, key="D0", text="Old Lore Page 1", title="Ancient Scroll"),
            ContextItem(index=1, key="D1", text="Target paragraph to translate", title="Ancient Scroll"),
            ContextItem(index=2, key="D2", text="Old Lore Page 3", title="Ancient Scroll"),
            ContextItem(index=3, key="D3", text="Old Lore Page 4", title="Ancient Scroll"),
        ]
        ctx = self.engine.build_context(doc, target_index=1, kind=ContextKind.DOCUMENT)
        self.assertIn("Title: Ancient Scroll", ctx.context_text)
        self.assertIn("Old Lore Page 1", ctx.context_text)
        self.assertIn("Old Lore Page 3", ctx.context_text)
        self.assertNotIn("Old Lore Page 4", ctx.context_text)  # Document only takes +-1

    def test_1500_char_limit_and_safe_boundary(self):
        # Create extremely long preceding line
        long_preceding = "Word " * 400  # ~2000 chars
        items = [
            ContextItem(index=0, key="A", text=long_preceding),
            ContextItem(index=1, key="B", text="Short target line"),
        ]
        ctx = self.engine.build_context(items, target_index=1, kind=ContextKind.SCENE)
        self.assertLessEqual(len(ctx.context_text), 1500)
        # Verify text does not end with broken UTF-8 or half word
        self.assertFalse(ctx.context_text.endswith(" "))

    def test_numeric_suffix_does_not_infer_speaker(self):
        # Dialogues with numeric suffix: QUEST_01_01, QUEST_01_02
        items = [
            ContextItem(index=0, key="QUEST_01_01", text="Hello hero"),
            ContextItem(index=1, key="QUEST_01_02", text="Farewell hero"),
        ]
        ctx = self.engine.build_context(items, target_index=1, kind=ContextKind.SCENE)
        # Context should not invent "Speaker 1" or "Speaker 2"
        self.assertNotIn("Speaker", ctx.context_text)
        self.assertNotIn("speaker", ctx.context_text.lower())

    def test_context_signature_excludes_paths_and_ordering_accidents(self):
        # Two identical contexts with different order in arbitrary non-semantic fields or path
        item = ContextItem(index=0, key="K1", text="Same line", document_path="/var/user/path/file.json")
        ctx1 = self.engine.build_context([item], target_index=0, kind=ContextKind.SCENE)

        item2 = ContextItem(index=0, key="K1", text="Same line", document_path="C:\\Users\\Game\\other.json")
        ctx2 = self.engine.build_context([item2], target_index=0, kind=ContextKind.SCENE)

        # Signature must be identical regardless of path
        self.assertEqual(ctx1.signature, ctx2.signature)
        self.assertNotIn("var", ctx1.signature)
        self.assertNotIn("Game", ctx1.signature)

    def test_same_source_different_context_distinct_signature(self):
        item_tavern = ContextItem(index=0, key="T1", text="Leave now.", scene_id="tavern_fight")
        ctx_tavern = self.engine.build_context([item_tavern], target_index=0, kind=ContextKind.SCENE)

        item_castle = ContextItem(index=0, key="C1", text="Leave now.", scene_id="royal_palace")
        ctx_castle = self.engine.build_context([item_castle], target_index=0, kind=ContextKind.SCENE)

        self.assertNotEqual(ctx_tavern.signature, ctx_castle.signature)


if __name__ == "__main__":
    unittest.main()
