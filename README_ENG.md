# Crimson Desert Localization Builder

[Tiếng Việt](README.md)

A Windows application for exporting, editing, translating, rebuilding, and
installing Crimson Desert localization data. The interface is served locally
through pywebview and does not download UI code from the Internet.

## Main features

- Unpack a localization package into one flat JSON file, including current game
  packages that contain multiple PALOC files.
- Repack JSON into the current package format using stable `marker + key`
  identities.
- Compare old and new catalogs to produce only new or changed source strings.
- Auto Translate with checkpoints, failure recovery, provider switching, and
  validation of placeholders, tags, newlines, and technical tokens.
- Local SQLite Translation Knowledge with translation memory, glossary,
  context, and import/export support.
- Unpack and rebuild the three required fonts in package `0012`.
- Install packages with backups, checksums, post-write verification, and
  rollback on failure.
- Uninstall/Restore from the latest applicable backup without changing unrelated
  packages.
- Responsive interface with four themes and detailed job status.

## Source languages

| Language | Package |
|---|---:|
| Korean | 0019 |
| English | 0020 |
| Japanese | 0021 |
| Russian | 0022 |
| Turkish | 0023 |
| Spanish | 0024 |
| Mexican Spanish | 0025 |
| French | 0026 |
| German | 0027 |
| Italian | 0028 |
| Polish | 0029 |
| Portuguese - Brazil | 0030 |
| Traditional Chinese | 0031 |
| Simplified Chinese | 0032 |

Changing the language automatically selects the matching package, JSON name,
PALOC entry, and font.

## Localization workflow

1. Select a language and its source package containing `0.pamt` and `0.paz`
   files.
2. Use **Unpack** to create the JSON catalog.
3. Edit only `translation`; preserve `marker`, `key`, `original`, and metadata.
4. Use **Repack** to create a new package in a separate output directory.

The application validates PAMT/PAZ structure, offsets, sizes, CRC values, and
checksums, then extracts the result again for final verification.

## Auto Translate

Auto Translate uses a three-step workflow:

1. Compare the previous translated JSON with the new source JSON and create a
   delta containing only new or source-changed strings.
2. Translate the delta with the selected provider. Completed rows are written
   to a checkpoint immediately, allowing pause, resume, or provider changes.
3. Validate structure and completeness, write repairs to a separate file when
   needed, and merge the delta with reusable previous translations.

Available choices:

- **AI Router (Free)**: selects among currently available keyless routes.
- **Google-free**.
- **Bing-free**, **Alibaba-free**, **ModernMT-free**, and **YouDao-free**.
- **Custom (OpenAI-compatible)**: user-defined HTTPS Base URL, model, API key,
  timeout, batch size, and concurrency.

Free providers may change quotas, rate limits, or availability. A Custom API key
is kept only for the current session by default. Enabling key persistence writes
it as plain text to `custom_openai_config.json`; do not share or commit that
file.

## Font package `0012`

The application unpacks and rebuilds exactly these three fonts:

```text
ui/basefont.ttf
ui/basefont_<language>.ttf
ui/creditfont.ttf
```

Keep the filenames unchanged when replacing fonts. The result is an `0012`
patch containing only `0.pamt` and the required PAZ file.

## Install and Restore

Install validates the input package, creates a manifest and SHA-256 backup,
copies the selected packages, updates related metadata, and reads the result
back for verification. An incomplete operation is rolled back from the backup.

Restore uses the most recent applicable Install backup and preserves metadata
for unrelated packages. Close the game before using Install or Restore.

## Requirements

- Windows 10 or Windows 11 x64.
- Microsoft Edge WebView2 Runtime.
- Python 3.13+ x64 for running or building from source.
- An Internet connection during build unless a Python runtime archive is
  supplied locally.

## Run from source

Open PowerShell in the repository root:

```powershell
python -m pip install --no-deps -r requirements-build.txt
python build_native.py
python app.py
```

`build_native.py` creates the checksum library at
`crimson0020\native\pa_checksum.dll`.

## Build the Windows one-folder package

The shortest build path is:

```powershell
.\BUILD_EXE.cmd
```

The script creates an isolated build environment and writes output to a new
`..\release_onedir_v2.0.0_<timestamp>` directory by default. Pass another output
directory as its first argument when needed:

```powershell
.\BUILD_EXE.cmd D:\builds\crimson-v2
```

Alternatively, invoke the builder directly and place output outside the source
tree:

```powershell
python scripts/build_onedir.py --output-root ..\build-output\onedir_candidate
```

To use an already downloaded Python runtime archive:

```powershell
python scripts/build_onedir.py `
  --output-root ..\build-output\onedir_candidate `
  --runtime-archive C:\path\to\python-3.14.7-amd64.zip
```

The builder verifies the locked runtime and dependencies, builds the native
checksum helper and launcher, runs a smoke test, and creates:

```text
onedir_candidate\
├── CrimsonDesertLocalizationBuilder\
│   ├── Crimson.Desert.Localization.Builder.exe
│   └── _internal\
└── CrimsonDesertLocalizationBuilder-windows-x64.zip
```

Keep `_internal` beside the executable and extract the complete ZIP before
launching it.

## Tests

```powershell
python -B scripts/verify_public_source.py .
python -B build_native.py
python -B -m unittest discover -s tests -v
python -B -m coverage run --source=crimson0020 -m unittest discover -s tests
python -B -m coverage report --fail-under=80
ruff check --no-cache crimson0020 cloudscraper scripts tests build_native.py build_launcher.py app.py sitecustomize.py proxy_tools
node --check crimson0020\webui\app.js
```

Default tests use synthetic/offline fixtures and do not call live providers.

## License

The source is released under the [MIT License](LICENSE). Dependency attribution
is listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
