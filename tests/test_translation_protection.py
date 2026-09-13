import unittest

from crimson0020.translation_runtime.contracts import FailureKind
from crimson0020.translation_runtime.protection import (
    ProtectionEngine,
    is_technical_bracket,
    mask_protected_content,
    restore_protected_content,
)


class TestTranslationProtection(unittest.TestCase):
    def test_repeated_br_tags(self):
        text = "Hello<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>World"
        masked = mask_protected_content(text)
        self.assertEqual(len(masked.tokens), 12)
        for i in range(12):
            self.assertIn(f"<x{i}>", masked.masked_text)
            self.assertEqual(masked.tokens[i], "<br/>")

        # Restoration
        translated_masked = masked.masked_text.replace("Hello", "Xin chào").replace("World", "Thế giới")
        restored, err = restore_protected_content(translated_masked, masked)
        self.assertIsNone(err)
        self.assertEqual(restored, "Xin chào<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>Thế giới")

    def test_repeated_identical_placeholders(self):
        text = "Item {0} has price {0} in store {0}."
        masked = mask_protected_content(text)
        self.assertEqual(len(masked.tokens), 3)
        self.assertEqual(masked.tokens, ("{0}", "{0}", "{0}"))
        self.assertEqual(masked.masked_text, "Item <x0> has price <x1> in store <x2>.")

        translated_masked = "Vật phẩm <x0> có giá <x1> tại cửa hàng <x2>."
        restored, err = restore_protected_content(translated_masked, masked)
        self.assertIsNone(err)
        self.assertEqual(restored, "Vật phẩm {0} có giá {0} tại cửa hàng {0}.")

    def test_nested_mixed_html_and_newlines(self):
        text = "Line 1\n<b>Bold <font color=\"#fff\">White</font></b>\r\nLine 2\\nEnd"
        masked = mask_protected_content(text)
        self.assertIn("\n", masked.tokens)
        self.assertIn("<b>", masked.tokens)
        self.assertIn('<font color="#fff">', masked.tokens)
        self.assertIn("</font>", masked.tokens)
        self.assertIn("</b>", masked.tokens)
        self.assertIn("\r\n", masked.tokens)
        self.assertIn("\\n", masked.tokens)

        # Exact round-trip
        restored, err = restore_protected_content(masked.masked_text, masked)
        self.assertIsNone(err)
        self.assertEqual(restored, text)

    def test_bracket_classification(self):
        # Technical brackets (must be masked)
        self.assertTrue(is_technical_bracket("[0]"))
        self.assertTrue(is_technical_bracket("[123]"))
        self.assertTrue(is_technical_bracket("[KEY_IDENTIFIER]"))
        self.assertTrue(is_technical_bracket("[ITEM_ID_01]"))
        self.assertTrue(is_technical_bracket("[color=#ff0000]"))
        self.assertTrue(is_technical_bracket("[/color]"))
        self.assertTrue(is_technical_bracket("[%s]"))
        self.assertTrue(is_technical_bracket("[$VAR]"))
        self.assertTrue(is_technical_bracket("[key:value]"))

        # Natural language brackets (must NOT be masked)
        self.assertFalse(is_technical_bracket("[Note: speak quietly]"))
        self.assertFalse(is_technical_bracket("[Tùy chọn nâng cao]"))
        self.assertFalse(is_technical_bracket("[Iron sword with sharp blade]"))
        self.assertFalse(is_technical_bracket("[Click here to continue]"))

        text = "Press [KEY_A] to attack [Goblin King] with [0] damage."
        masked = mask_protected_content(text)
        # [KEY_A] and [0] are technical -> masked
        # [Goblin King] is natural language -> kept in masked_text for translation
        self.assertIn("[Goblin King]", masked.masked_text)
        self.assertNotIn("[KEY_A]", masked.masked_text)
        self.assertNotIn("[0]", masked.masked_text)

    def test_unicode_and_empty_text(self):
        empty_masked = mask_protected_content("")
        self.assertEqual(empty_masked.masked_text, "")
        self.assertEqual(empty_masked.tokens, ())
        restored, err = restore_protected_content("", empty_masked)
        self.assertEqual(restored, "")
        self.assertIsNone(err)

        unicode_text = "Chào mừng 🎮 tới xứ sở tuyết ❄️: {player_name}!"
        masked = mask_protected_content(unicode_text)
        self.assertEqual(masked.tokens, ("{player_name}",))
        self.assertIn("🎮", masked.masked_text)
        self.assertIn("❄️", masked.masked_text)

    def test_missing_token_detection(self):
        text = "Damage: {0}, Defense: {1}"
        masked = mask_protected_content(text)
        # Provider omitted <x1>
        bad_translation = "Sát thương: <x0>, Phòng thủ:"
        restored, err = restore_protected_content(bad_translation, masked)
        self.assertIsNotNone(err)
        self.assertEqual(err.kind, FailureKind.PROTECTED_CONTENT_VIOLATION)
        self.assertIn("missing", err.message.lower())

    def test_duplicate_token_detection(self):
        text = "Level {0}"
        masked = mask_protected_content(text)
        bad_translation = "Cấp độ <x0> và <x0>"
        restored, err = restore_protected_content(bad_translation, masked)
        self.assertIsNotNone(err)
        self.assertEqual(err.kind, FailureKind.PROTECTED_CONTENT_VIOLATION)
        self.assertIn("duplicate", err.message.lower())

    def test_reordered_token_detection(self):
        text = "From {0} to {1}"
        masked = mask_protected_content(text)
        # Reordered tokens <x1> before <x0>
        reordered = "Từ <x1> đến <x0>"
        restored, err = restore_protected_content(reordered, masked)
        self.assertIsNotNone(err)
        self.assertEqual(err.kind, FailureKind.PROTECTED_CONTENT_VIOLATION)
        self.assertIn("reordered", err.message.lower())

    def test_fake_malicious_token_injection(self):
        text = "Simple text without tokens"
        masked = mask_protected_content(text)
        # Provider hallucinated or injected a fake token <x0>
        bad_translation = "Văn bản với mã độc <x0>"
        restored, err = restore_protected_content(bad_translation, masked)
        self.assertIsNotNone(err)
        self.assertEqual(err.kind, FailureKind.PROTECTED_CONTENT_VIOLATION)
        self.assertIn("unexpected", err.message.lower())

    def test_mutation_of_token(self):
        text = "Value: {0}"
        masked = mask_protected_content(text)
        # Token mutated to <X0> or <x 0>
        mutated = "Giá trị: <X0>"
        restored, err = restore_protected_content(mutated, masked)
        self.assertIsNotNone(err)
        self.assertEqual(err.kind, FailureKind.PROTECTED_CONTENT_VIOLATION)

    def test_protection_engine_facade(self):
        m = ProtectionEngine.mask("Hello {0}")
        self.assertEqual(len(m.tokens), 1)
        r, err = ProtectionEngine.restore("Xin chào <x0>", m)
        self.assertIsNone(err)
        self.assertEqual(r, "Xin chào {0}")


if __name__ == "__main__":
    unittest.main()
