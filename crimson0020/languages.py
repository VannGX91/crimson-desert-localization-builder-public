from dataclasses import dataclass


LANGUAGE_MAP = {
    "English": ("0020", "localizationstring_eng.paloc", "basefont_eng.ttf"),
    "Korean": ("0019", "localizationstring_kor.paloc", "basefont_kor.ttf"),
    "Japanese": ("0021", "localizationstring_jpn.paloc", "basefont_jpn.ttf"),
    "Russian": ("0022", "localizationstring_rus.paloc", "basefont_rus.ttf"),
    "Turkish": ("0023", "localizationstring_tur.paloc", "basefont_tur.ttf"),
    "Spanish": ("0024", "localizationstring_spa-es.paloc", "basefont_spa-es.ttf"),
    "Mexican Spanish": (
        "0025",
        "localizationstring_spa-mx.paloc",
        "basefont_spa-mx.ttf",
    ),
    "French": ("0026", "localizationstring_fre.paloc", "basefont_fre.ttf"),
    "German": ("0027", "localizationstring_ger.paloc", "basefont_ger.ttf"),
    "Italian": ("0028", "localizationstring_ita.paloc", "basefont_ita.ttf"),
    "Polish": ("0029", "localizationstring_pol.paloc", "basefont_pol.ttf"),
    "Portuguese - Brazil": (
        "0030",
        "localizationstring_por-br.paloc",
        "basefont_por-br.ttf",
    ),
    "Traditional Chinese": (
        "0031",
        "localizationstring_zho-tw.paloc",
        "basefont_zho-tw.ttf",
    ),
    "Simplified Chinese": (
        "0032",
        "localizationstring_zho-cn.paloc",
        "basefont_zho-cn.ttf",
    ),
}


@dataclass(frozen=True)
class LanguageSpec:
    name: str
    package_folder: str
    paloc_filename: str
    font_filename: str

    @property
    def paloc_entry_path(self) -> str:
        return f"gamedata/{self.paloc_filename}"

    @property
    def json_filename(self) -> str:
        return f"{self.paloc_filename}.json"

    @property
    def font_entry_path(self) -> str:
        return f"ui/{self.font_filename}"

    @property
    def font_entry_paths(self) -> tuple[str, str, str]:
        return (
            "ui/basefont.ttf",
            self.font_entry_path,
            "ui/creditfont.ttf",
        )


def get_language_spec(language_name: str) -> LanguageSpec:
    try:
        package_folder, paloc_filename, font_filename = LANGUAGE_MAP[language_name]
    except KeyError as error:
        raise ValueError(f"Unsupported language: {language_name}") from error
    return LanguageSpec(
        name=language_name,
        package_folder=package_folder,
        paloc_filename=paloc_filename,
        font_filename=font_filename,
    )
