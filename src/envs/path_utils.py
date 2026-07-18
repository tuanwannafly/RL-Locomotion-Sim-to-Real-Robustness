"""Resolve Windows long paths to 8.3 short paths. Works around a MuJoCo
Windows bug where ``MjModel.from_xml_path`` fails on paths containing
non-ASCII characters (e.g. the em-dash in this workspace's name).

Reference: https://github.com/google-deepmind/mujoco/issues/3011
"""
from __future__ import annotations

import atexit
import ctypes
import os
import shutil
import tempfile

# ---------------------------------------------------------------------------
# 1. ``short_path`` — convert any existing path to its 8.3 short form.
# ---------------------------------------------------------------------------

if os.name == "nt":
    _kernel32 = ctypes.windll.kernel32
    _GetShortPathNameW = _kernel32.GetShortPathNameW
    _GetShortPathNameW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
    _GetShortPathNameW.restype = ctypes.c_uint32


def short_path(long_path: str) -> str:
    """Return the Windows 8.3 short path for *long_path*.

    Falls back to the original path if conversion fails or is unavailable
    (e.g. on non-Windows platforms, or when short-name generation is off)."""
    if not long_path or not os.path.exists(long_path):
        return long_path
    if os.name != "nt":
        return long_path
    buf = ctypes.create_unicode_buffer(1024)
    n = _GetShortPathNameW(long_path, buf, 1024)
    return buf.value if n else long_path


# ---------------------------------------------------------------------------
# 2. Asset directory shadowing for gymnasium mujoco envs.
#
# Each `gym.make('Ant-v4')` calls ``expand_model_path('ant.xml')`` which
# joins ``<gymnasium_install>/envs/mujoco/assets/ant.xml``. If that
# directory contains non-ASCII characters, the C-side ``MjModel.from_xml_path``
# fails to open the file. We solve this by **replacing** the assets
# directory that ``expand_model_path`` resolves to with a temporary
# directory whose path is ASCII-only.
# ---------------------------------------------------------------------------

_shadow_state: dict = {"tempdir": None, "patched": False}


def _shadow_gymnasium_assets() -> None:
    if _shadow_state["patched"]:
        return

    import gymnasium.envs.mujoco.mujoco_env as _me
    from os import path as _path

    pkg_assets_dir = _path.join(_path.dirname(_me.__file__), "assets")
    if not _path.isdir(pkg_assets_dir):
        return

    # Build an ASCII-only temp dir.
    base_temp = tempfile.gettempdir()
    safe_temp = "".join(c if ord(c) < 128 else "_" for c in base_temp)
    _shadow_state["tempdir"] = tempfile.TemporaryDirectory(
        prefix="gym_mj_assets_", dir=safe_temp
    )
    temp_assets = _shadow_state["tempdir"].name

    # Copy each asset (file or subdir) to the temp location.
    for entry in os.listdir(pkg_assets_dir):
        s = _path.join(pkg_assets_dir, entry)
        d = _path.join(temp_assets, entry)
        if _path.isfile(s):
            shutil.copy2(s, d)
        elif _path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)

    _original_expand = _me.expand_model_path

    def _patched_expand_model_path(model_path: str) -> str:
        fullpath = _original_expand(model_path)
        filename = os.path.basename(fullpath)
        ascii_candidate = os.path.join(temp_assets, filename)
        if os.path.exists(ascii_candidate):
            return ascii_candidate
        return short_path(fullpath)

    _me.expand_model_path = _patched_expand_model_path
    _shadow_state["patched"] = True

    atexit.register(_cleanup_shadow)


def _cleanup_shadow() -> None:
    if _shadow_state["tempdir"] is not None:
        _shadow_state["tempdir"].cleanup()
        _shadow_state["tempdir"] = None
        _shadow_state["patched"] = False


def patch_gym_make() -> None:
    """Apply the path workaround. Call once at program startup before any
    ``gym.make('*-v4')`` or ``gym.make('*-v5')`` call.

    Idempotent. No-op on non-Windows or when the working directory is
    already pure-ASCII."""
    if os.name != "nt":
        return
    cwd = os.getcwd()
    if all(ord(c) < 128 for c in cwd):
        return
    _shadow_gymnasium_assets()