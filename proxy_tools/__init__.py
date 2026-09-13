# ruff: noqa: E731
"""Compatibility dependency for pywebview 6.2.1.

Vendored from proxy_tools 0.1.0. The upstream package is MIT licensed; see
THIRD_PARTY_NOTICES.md. Keeping this tiny pure-Python dependency as readable
source avoids executing an sdist build during deterministic release packaging.
"""

import sys


PY2 = sys.version_info[0] == 2


def _identity(value):
    return value


if PY2:
    def implements_bool(cls):
        cls.__nonzero__ = cls.__bool__
        del cls.__bool__
        return cls
else:
    implements_bool = _identity


@implements_bool
class Proxy:
    __slots__ = ("__local", "__dict__", "__name__")

    def __init__(self, local, name=None):
        object.__setattr__(self, "_Proxy__local", local)
        object.__setattr__(self, "__name__", name)

    def _get_current_object(self):
        if not hasattr(self.__local, "__release_local__"):
            return self.__local()
        try:
            return getattr(self.__local, self.__name__)
        except AttributeError as error:
            raise RuntimeError(f"no object bound to {self.__name__}") from error

    @property
    def __dict__(self):
        try:
            return self._get_current_object().__dict__
        except RuntimeError as error:
            raise AttributeError("__dict__") from error

    def __repr__(self):
        try:
            return repr(self._get_current_object())
        except RuntimeError:
            return f"<{self.__class__.__name__} unbound>"

    def __bool__(self):
        try:
            return bool(self._get_current_object())
        except RuntimeError:
            return False

    def __dir__(self):
        try:
            return dir(self._get_current_object())
        except RuntimeError:
            return []

    def __getattr__(self, name):
        if name == "__members__":
            return dir(self._get_current_object())
        return getattr(self._get_current_object(), name)

    __setattr__ = lambda self, name, value: setattr(self._get_current_object(), name, value)
    __delattr__ = lambda self, name: delattr(self._get_current_object(), name)
    __str__ = lambda self: str(self._get_current_object())
    __hash__ = lambda self: hash(self._get_current_object())
    __call__ = lambda self, *args, **kwargs: self._get_current_object()(*args, **kwargs)
    __len__ = lambda self: len(self._get_current_object())
    __getitem__ = lambda self, key: self._get_current_object()[key]
    __setitem__ = lambda self, key, value: self._get_current_object().__setitem__(key, value)
    __delitem__ = lambda self, key: self._get_current_object().__delitem__(key)
    __iter__ = lambda self: iter(self._get_current_object())
    __contains__ = lambda self, value: value in self._get_current_object()
    __lt__ = lambda self, other: self._get_current_object() < other
    __le__ = lambda self, other: self._get_current_object() <= other
    __eq__ = lambda self, other: self._get_current_object() == other
    __ne__ = lambda self, other: self._get_current_object() != other
    __gt__ = lambda self, other: self._get_current_object() > other
    __ge__ = lambda self, other: self._get_current_object() >= other
    __add__ = lambda self, other: self._get_current_object() + other
    __sub__ = lambda self, other: self._get_current_object() - other
    __mul__ = lambda self, other: self._get_current_object() * other
    __floordiv__ = lambda self, other: self._get_current_object() // other
    __truediv__ = lambda self, other: self._get_current_object() / other
    __mod__ = lambda self, other: self._get_current_object() % other
    __pow__ = lambda self, other: self._get_current_object() ** other
    __neg__ = lambda self: -self._get_current_object()
    __pos__ = lambda self: +self._get_current_object()
    __abs__ = lambda self: abs(self._get_current_object())
    __invert__ = lambda self: ~self._get_current_object()
    __int__ = lambda self: int(self._get_current_object())
    __float__ = lambda self: float(self._get_current_object())
    __index__ = lambda self: self._get_current_object().__index__()
    __enter__ = lambda self: self._get_current_object().__enter__()
    __exit__ = lambda self, *args: self._get_current_object().__exit__(*args)
    __radd__ = lambda self, other: other + self._get_current_object()
    __rsub__ = lambda self, other: other - self._get_current_object()
    __rmul__ = lambda self, other: other * self._get_current_object()
    __rtruediv__ = lambda self, other: other / self._get_current_object()


module_property = Proxy
proxy = Proxy
