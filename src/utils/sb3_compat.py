from __future__ import annotations

import importlib
import sys


def _normalize_bit_generator_name(value) -> str:
    if isinstance(value, str):
        text = value.strip()
        # Handle repr-style class strings:
        # "<class 'numpy.random._pcg64.PCG64'>" -> "PCG64"
        if text.startswith("<class '") and text.endswith("'>"):
            text = text[len("<class '") : -2]
        if "." in text:
            text = text.split(".")[-1]
        return text
    if isinstance(value, type):
        return value.__name__
    return str(value)


def install_numpy_pickle_compat_shims() -> None:
    """Install aliases for numpy private module paths used in pickled SB3 data."""
    try:
        import numpy.core as numpy_core
    except Exception:
        return

    sys.modules.setdefault("numpy._core", numpy_core)

    core_submodules = [
        "numeric",
        "multiarray",
        "umath",
        "_multiarray_umath",
        "fromnumeric",
        "shape_base",
        "arrayprint",
        "getlimits",
        "overrides",
    ]
    for name in core_submodules:
        try:
            mod = importlib.import_module(f"numpy.core.{name}")
        except Exception:
            continue
        sys.modules.setdefault(f"numpy._core.{name}", mod)

    # Compatibility for older numpy RNG pickles that may pass class objects
    # or repr-style class strings into numpy.random._pickle.__bit_generator_ctor.
    try:
        import numpy.random._pickle as np_pickle
    except Exception:
        return

    ctor = getattr(np_pickle, "__bit_generator_ctor", None)
    bit_generators = getattr(np_pickle, "BitGenerators", None)
    if ctor is None or not isinstance(bit_generators, dict):
        return

    if getattr(np_pickle, "_codex_bitgen_ctor_patched", False):
        return

    def _compat_bit_generator_ctor(bit_generator_name="MT19937"):
        key = _normalize_bit_generator_name(bit_generator_name)
        if key in bit_generators:
            return bit_generators[key]()
        return ctor(bit_generator_name)

    np_pickle.__bit_generator_ctor = _compat_bit_generator_ctor
    np_pickle._codex_bitgen_ctor_patched = True


def make_legacy_custom_objects(observation_space=None, action_space=None) -> dict:
    """Custom objects to load SB3 zips saved with older dependency versions."""

    def _constant_schedule(_progress_remaining: float) -> float:
        return 0.0

    custom_objects: dict = {
        "lr_schedule": _constant_schedule,
        "clip_range": _constant_schedule,
        "clip_range_vf": _constant_schedule,
    }
    if observation_space is not None:
        custom_objects["observation_space"] = observation_space
    if action_space is not None:
        custom_objects["action_space"] = action_space
    return custom_objects
