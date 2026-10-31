# Crimson Desert Localization Builder

makasih sama mas sioren.
[English](README_ENG.md)

Công cụ Windows hỗ trợ xuất, chỉnh sửa, dịch, đóng gói và cài đặt dữ liệu
localization của Crimson Desert. Ứng dụng sử dụng giao diện pywebview cục bộ và
không tải mã giao diện từ Internet.

## Tính năng chính

- Unpack package localization thành một file JSON phẳng, kể cả package game mới
  có nhiều PALOC.
- Repack JSON vào package hiện tại bằng định danh ổn định `marker + key`.
- So sánh catalog cũ và mới để chỉ tạo danh sách câu mới hoặc đã thay đổi.
- Auto Translate có checkpoint, tiếp tục sau lỗi, đổi provider giữa chừng và
  kiểm tra placeholder, tag, newline cùng token kỹ thuật.
- Translation Knowledge cục bộ bằng SQLite, gồm translation memory, glossary,
  context và import/export.
- Unpack và repack ba font cần thiết trong package `0012`.
- Install package vào game với backup, checksum, xác minh sau ghi và rollback
  khi có lỗi.
- Uninstall/Restore từ backup gần nhất mà không làm thay đổi package khác.
- Giao diện responsive với bốn theme và bảng trạng thái chi tiết.

## Ngôn ngữ nguồn

| Ngôn ngữ | Package |
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

Khi đổi ngôn ngữ, tool tự chọn package, tên JSON, PALOC và font tương ứng.

## Quy trình localization

1. Chọn ngôn ngữ và package nguồn có `0.pamt` cùng các file `0.paz`.
2. Chọn **Unpack** để tạo JSON.
3. Chỉ chỉnh trường `translation`; giữ nguyên `marker`, `key`, `original` và
   metadata.
4. Chọn **Repack** để tạo package mới trong thư mục output riêng.

Tool kiểm tra cấu trúc PAMT/PAZ, offset, kích thước, CRC, checksum và giải nén
lại kết quả để xác minh trước khi bàn giao.

## Auto Translate

Auto Translate sử dụng quy trình ba bước:

1. So sánh JSON dịch cũ với JSON nguồn mới để tạo delta chỉ gồm câu mới hoặc
   câu có nguồn thay đổi.
2. Dịch delta bằng provider đã chọn. Kết quả hoàn tất được ghi checkpoint ngay
   để có thể tạm dừng, tiếp tục hoặc chuyển provider.
3. Kiểm tra cấu trúc và độ đầy đủ, sửa cấu trúc sang file mới nếu cần, sau đó
   ghép với bản dịch cũ để tạo JSON hoàn chỉnh.

Các lựa chọn hiện có:

- **AI Router (Free)**: tự chọn giữa các route không cần key đang khả dụng.
- **Google-free**.
- **Bing-free**, **Alibaba-free**, **ModernMT-free** và **YouDao-free**.
- **Custom (OpenAI-compatible)**: Base URL HTTPS, model, API key, timeout,
  batch và concurrency do người dùng cấu hình.

Provider miễn phí có thể thay đổi quota, giới hạn tốc độ hoặc ngừng phản hồi.
API key Custom mặc định chỉ tồn tại trong phiên chạy. Nếu bật tùy chọn lưu key,
key sẽ được ghi dạng text vào `custom_openai_config.json`; không chia sẻ hoặc
commit file này.

## Font `0012`

Tool unpack và repack đúng ba font:

```text
ui/basefont.ttf
ui/basefont_<language>.ttf
ui/creditfont.ttf
```

Giữ nguyên tên file khi thay font. Kết quả là patch package `0012` chỉ chứa
`0.pamt` và PAZ cần thiết.

## Install và Restore

Install xác minh package đầu vào, tạo backup có manifest và SHA-256, chép package
đã chọn, cập nhật metadata liên quan rồi đọc lại toàn bộ kết quả. Nếu thao tác
không hoàn tất, tool phục hồi từ backup.

Restore sử dụng backup Install chưa phục hồi gần nhất và giữ nguyên metadata của
các package không thuộc lần cài đó. Nên đóng game trước khi Install hoặc Restore.

## Yêu cầu hệ thống

- Windows 10 hoặc Windows 11 x64.
- Microsoft Edge WebView2 Runtime.
- Python 3.13+ x64 khi chạy hoặc build từ source.
- Kết nối Internet khi build nếu chưa cung cấp sẵn Python runtime archive.

## Chạy từ source

Mở PowerShell tại thư mục repository:

```powershell
python -m pip install --no-deps -r requirements-build.txt
python build_native.py
python app.py
```

`build_native.py` tạo thư viện checksum tại
`crimson0020\native\pa_checksum.dll`.

## Build bản Windows one-folder

Cách nhanh nhất là chạy:

```powershell
.\BUILD_EXE.cmd
```

Script tạo môi trường build riêng và mặc định ghi kết quả vào thư mục mới dạng
`..\release_onedir_v2.0.0_<thời-gian>`. Có thể truyền thư mục output khác làm
tham số đầu tiên:

```powershell
.\BUILD_EXE.cmd D:\builds\crimson-v2
```

Hoặc gọi builder trực tiếp và đặt output bên ngoài cây source:

```powershell
python scripts/build_onedir.py --output-root ..\build-output\onedir_candidate
```

Để dùng Python runtime archive đã tải sẵn:

```powershell
python scripts/build_onedir.py `
  --output-root ..\build-output\onedir_candidate `
  --runtime-archive C:\path\to\python-3.14.7-amd64.zip
```

Builder xác minh runtime và dependency đã khóa, build native checksum cùng
launcher, chạy smoke test và tạo:

```text
onedir_candidate\
├── CrimsonDesertLocalizationBuilder\
│   ├── Crimson.Desert.Localization.Builder.exe
│   └── _internal\
└── CrimsonDesertLocalizationBuilder-windows-x64.zip
```

Phải giữ `_internal` cạnh EXE và giải nén toàn bộ ZIP trước khi chạy.

## Kiểm thử

```powershell
python -B scripts/verify_public_source.py .
python -B build_native.py
python -B -m unittest discover -s tests -v
python -B -m coverage run --source=crimson0020 -m unittest discover -s tests
python -B -m coverage report --fail-under=80
ruff check --no-cache crimson0020 cloudscraper scripts tests build_native.py build_launcher.py app.py sitecustomize.py proxy_tools
node --check crimson0020\webui\app.js
```

Các test mặc định dùng fixture synthetic/offline và không gọi provider live.

## Giấy phép

Source được phát hành theo [MIT License](LICENSE). Attribution của dependency
được ghi trong [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
