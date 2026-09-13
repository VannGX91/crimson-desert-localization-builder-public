"""Fail-closed compatibility surface for the optional ``translators`` client.

``translators`` imports ``cloudscraper`` unconditionally even when it uses its
normal ``requests`` transport.  The full third-party cloudscraper dependency
executes untrusted JavaScript through Js2Py when solving a web challenge.  This
application never requests that transport, so only the import surface is kept
and every attempted use fails explicitly.
"""

from typing import NoReturn


_DISABLED_MESSAGE = (
    "The cloudscraper transport is disabled; use the requests transport instead."
)


def create_scraper(*_args: object, **_kwargs: object) -> NoReturn:
    raise RuntimeError(_DISABLED_MESSAGE)


def create_async_scraper(*_args: object, **_kwargs: object) -> NoReturn:
    raise RuntimeError(_DISABLED_MESSAGE)


__all__ = ("create_async_scraper", "create_scraper")
