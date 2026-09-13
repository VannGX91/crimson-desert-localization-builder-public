import unittest

from crimson0020.languages import LANGUAGE_MAP, get_language_spec


EXPECTED_LANGUAGE_MAP = {
    "English": ("0020", "localizationstring_eng.paloc", "basefont_eng.ttf"),
    "Korean": ("0019", "localizationstring_kor.paloc", "basefont_kor.ttf"),
    "Japanese": ("0021", "localizationstring_jpn.paloc", "basefont_jpn.ttf"),
    "Russian": ("0022", "localizationstring_rus.paloc", "basefont_rus.ttf"),
    "Turkish": ("0023", "localizationstring_tur.paloc", "basefont_tur.ttf"),
    "Spanish": ("0024", "localizationstring_spa-es.paloc", "basefont_spa-es.ttf"),
    "Mexican Spanish": ("0025", "localizationstring_spa-mx.paloc", "basefont_spa-mx.ttf"),
    "French": ("0026", "localizationstring_fre.paloc", "basefont_fre.ttf"),
    "German": ("0027", "localizationstring_ger.paloc", "basefont_ger.ttf"),
    "Italian": ("0028", "localizationstring_ita.paloc", "basefont_ita.ttf"),
    "Polish": ("0029", "localizationstring_pol.paloc", "basefont_pol.ttf"),
    "Portuguese - Brazil": ("0030", "localizationstring_por-br.paloc", "basefont_por-br.ttf"),
    "Traditional Chinese": ("0031", "localizationstring_zho-tw.paloc", "basefont_zho-tw.ttf"),
    "Simplified Chinese": ("0032", "localizationstring_zho-cn.paloc", "basefont_zho-cn.ttf"),
}


class LanguageTests(unittest.TestCase):
    def test_language_map_matches_the_supported_game_packages(self) -> None:
        self.assertEqual(EXPECTED_LANGUAGE_MAP, LANGUAGE_MAP)

    def test_spec_derives_archive_json_and_font_paths(self) -> None:
        spec = get_language_spec("Mexican Spanish")

        self.assertEqual("0025", spec.package_folder)
        self.assertEqual(
            "gamedata/localizationstring_spa-mx.paloc",
            spec.paloc_entry_path,
        )
        self.assertEqual(
            "localizationstring_spa-mx.paloc.json",
            spec.json_filename,
        )
        self.assertEqual("ui/basefont_spa-mx.ttf", spec.font_entry_path)
        self.assertEqual(
            ("ui/basefont.ttf", "ui/basefont_spa-mx.ttf", "ui/creditfont.ttf"),
            spec.font_entry_paths,
        )

    def test_unknown_language_is_rejected_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported language"):
            get_language_spec("Vietnamese")


if __name__ == "__main__":
    unittest.main()
