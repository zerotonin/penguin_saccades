# ╔══════════════════════════════════════════════════════════════════╗
# ║  penguin_saccades — config                                       ║
# ║  « one resolver for every machine-specific path »                ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  No absolute path is hard-coded anywhere in the package. Paths   ║
# ║  flow through the gitignored local_paths.json, resolved against  ║
# ║  the committed local_paths.template.json. Resolution order for   ║
# ║  the data root: 1. PENGUIN_SACCADES_DATA_ROOT environment        ║
# ║  variable 2. data_root of the active profile in                  ║
# ║  local_paths.json 3. in-repo data/ symlink A missing             ║
# ║  local_paths.json fails loudly, naming the template.             ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Machine-specific path resolution."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_PATHS_FILE: Path = _REPO_ROOT / "local_paths.json"
_TEMPLATE_FILE: Path = _REPO_ROOT / "local_paths.template.json"
_DATA_SYMLINK: Path = _REPO_ROOT / "data"

ENV_DATA_ROOT: str = "PENGUIN_SACCADES_DATA_ROOT"
ENV_PROFILE: str = "PENGUIN_SACCADES_PROFILE"
ENV_PATHS_FILE: str = "PENGUIN_SACCADES_PATHS_FILE"


class PathConfigError(RuntimeError):
    """Raised when machine-specific configuration cannot be resolved."""


# ─────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────
def data_root() -> Path:
    """Resolve the project data root (env, then JSON, then in-repo symlink)."""
    env_value = os.environ.get(ENV_DATA_ROOT)
    if env_value:
        return Path(env_value).expanduser()
    profile_value = active_profile().get("data_root")
    if profile_value:
        return Path(profile_value).expanduser()
    if _DATA_SYMLINK.exists():
        return _DATA_SYMLINK
    raise PathConfigError(
        f"No data root configured.  Set ${ENV_DATA_ROOT}, add 'data_root' to the "
        f"active profile in {_paths_file().name}, or create a {_DATA_SYMLINK.name}/ symlink."
    )


def get_path(key: str) -> Path:
    """Return a profile key as an expanded ``Path``; fail loudly if unset."""
    value = active_profile().get(key)
    if not value:
        raise PathConfigError(
            f"Key '{key}' is not set in profile '{active_profile_name()}' of "
            f"{_paths_file().name}.  See {_TEMPLATE_FILE.name} for the expected keys."
        )
    return Path(value).expanduser()


def output_dir() -> Path:
    """Results directory: profile ``output_dir`` or ``data_root()/results``."""
    try:
        return get_path("output_dir")
    except PathConfigError:
        return data_root() / "results"


def active_profile_name() -> str:
    """Name of the active profile (env, then JSON ``active_profile``, then ``local``)."""
    return os.environ.get(ENV_PROFILE) or _config().get("active_profile", "local")


def active_profile() -> dict[str, str]:
    """Return the active profile dictionary."""
    profiles = _config().get("profiles", {})
    name = active_profile_name()
    if name not in profiles:
        raise PathConfigError(
            f"Profile '{name}' not found in {_paths_file().name}.  "
            f"Available: {', '.join(profiles) or '<none>'}."
        )
    return profiles[name]


# ─────────────────────────────────────────────────────────────────
#  Internals
# ─────────────────────────────────────────────────────────────────
def _paths_file() -> Path:
    override = os.environ.get(ENV_PATHS_FILE)
    return Path(override).expanduser() if override else _PATHS_FILE


def _config() -> dict:
    path = _paths_file()
    if not path.exists():
        raise PathConfigError(
            f"{path} is missing.  Copy {_TEMPLATE_FILE.name} to {path.name} in the "
            "repository root and fill in the paths for this machine."
        )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _export_shell() -> str:
    """Render the active profile as ``export KEY=value`` lines for bash."""
    lines = [f"export PENGUIN_SACCADES_PROFILE={active_profile_name()}"]
    for key, value in active_profile().items():
        lines.append(f"export PENGUIN_SACCADES_{key.upper()}='{value}'")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """``python -m penguin_saccades.config --export`` for shell consumers."""
    parser = argparse.ArgumentParser(description="Resolve machine-specific paths.")
    parser.add_argument("--export", action="store_true", help="print bash export lines")
    parser.add_argument("--profile", default=None, help="override the active profile")
    args = parser.parse_args(argv)
    if args.profile:
        os.environ[ENV_PROFILE] = args.profile
    try:
        if args.export:
            print(_export_shell())
        else:
            print(f"profile: {active_profile_name()}")
            print(f"data_root: {data_root()}")
            print(f"output_dir: {output_dir()}")
    except PathConfigError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
