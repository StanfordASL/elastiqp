"""ElastiQP: an always-feasible QP solver for constrained robot control"""


def _import_core():  # compiled nanobind core
    import os

    # Let ELASTIQP_BUILD_DIR take priority for ctest
    env_dir = os.environ.get("ELASTIQP_BUILD_DIR")
    if env_dir:
        import importlib.machinery
        import importlib.util
        import sys

        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            path = os.path.join(env_dir, "_core" + suffix)
            if not os.path.exists(path):
                continue
            spec = importlib.util.spec_from_file_location("elastiqp._core", path)
            core = importlib.util.module_from_spec(spec)
            sys.modules["elastiqp._core"] = core
            spec.loader.exec_module(core)
            return core
    try:
        from elastiqp import _core

        return _core
    except ImportError:
        pass
    # Un-configured source checkout: the extension defaults to <repo>/build.
    from pathlib import Path

    build = str(Path(__file__).resolve().parents[2] / "build")
    if os.path.isdir(build) and build not in __path__:
        __path__.append(build)
    try:
        from elastiqp import _core
    except ImportError as e:
        raise ImportError(
            "elastiqp._core extension not found. Install the package "
            "(pip install .) or build from source (cmake -B build . && "
            "cmake --build build), optionally pointing ELASTIQP_BUILD_DIR "
            "at the build directory."
        ) from e
    return _core


_core = _import_core()

IpmSettings = _core.IpmSettings
IpmSolver = _core.IpmSolver
Settings = _core.Settings
Solution = _core.Solution
Solver = _core.Solver
Status = _core.Status
solve = _core.solve

__all__ = [
    "IpmSettings",
    "IpmSolver",
    "Settings",
    "Solution",
    "Solver",
    "Status",
    "solve",
]

try:
    from importlib.metadata import version as _version

    __version__ = _version("elastiqp")
except Exception:  # not installed (source checkout)
    __version__ = "0.0.0.dev0"

del _core, _import_core
