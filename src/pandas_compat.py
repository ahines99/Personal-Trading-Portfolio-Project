"""Pandas compatibility shim for old pickle caches.

Older pandas versions pickled StringDtype with a `na_value` argument that
the current pandas (2.2+) no longer accepts. This shim monkey-patches
StringDtype.__init__ to silently drop the deprecated argument so old
caches deserialize cleanly.

Import this module ONCE at the very top of the entry point (run_strategy.py,
run_phase_d_bisect.py, etc.) BEFORE any pickle.load calls.
"""
from pandas.core.arrays.string_ import StringDtype

_orig_init = StringDtype.__init__

def _compat_init(self, *args, **kwargs):
    # Drop deprecated na_value kwarg
    kwargs.pop("na_value", None)
    # Old positional format: (storage, na_value) — keep only storage
    if len(args) >= 2:
        args = args[:1]
    _orig_init(self, *args, **kwargs)

StringDtype.__init__ = _compat_init
