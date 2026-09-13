import ctypes
import os
from pathlib import Path


_LIBRARY: ctypes.CDLL | None = None
_LOAD_ATTEMPTED = False


def _load_library() -> ctypes.CDLL | None:
    global _LIBRARY, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _LIBRARY
    _LOAD_ATTEMPTED = True
    if os.name != "nt":
        return None
    library_path = Path(__file__).resolve().parent / "native" / "pa_checksum.dll"
    if not library_path.is_file():
        return None
    try:
        library = ctypes.CDLL(str(library_path))
    except OSError:
        return None
    library.pa_checksum_file_w.argtypes = (
        ctypes.c_wchar_p,
        ctypes.POINTER(ctypes.c_int),
    )
    library.pa_checksum_file_w.restype = ctypes.c_uint32
    _LIBRARY = library
    return library


def native_pa_checksum_file(path: Path | str) -> int | None:
    library = _load_library()
    if library is None:
        return None
    error_code = ctypes.c_int(0)
    checksum = library.pa_checksum_file_w(
        str(Path(path).resolve()),
        ctypes.byref(error_code),
    )
    if error_code.value:
        raise OSError(error_code.value, os.strerror(error_code.value), str(path))
    return int(checksum)


def native_checksum_available() -> bool:
    return _load_library() is not None
