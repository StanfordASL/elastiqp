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


import sys as _sys

_core = _import_core()

Solution = _core.Solution
Status = _core.Status
solve = _core.solve

# One submodule per backend, mirroring the C++ namespaces: each has its own
# Settings and Solver (das.RowState too). Registered in sys.modules so that
# `import elastiqp.das` works as well as `elastiqp.das`.
das = _core.das
pdal = _core.pdal
ipm = _core.ipm
for _name, _mod in (("das", das), ("pdal", pdal), ("ipm", ipm)):
    _sys.modules[f"elastiqp.{_name}"] = _mod
RowState = das.RowState

METHODS = ("das", "pdal", "ipm")
_BACKENDS = {"das": das, "pdal": pdal, "ipm": ipm}


def _lookup(table, method):
    try:
        return table[method]
    except KeyError:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}") from None


def Solver(method="das"):
    """Factory: a solver object for the given backend.

    method is "das" (dual active set, the default), "pdal" (primal-dual
    augmented Lagrangian) or "ipm" (interior point); the returned object is
    an elastiqp.das.Solver / pdal.Solver / ipm.Solver. All three share
    setup(), update(), set_*(), solve() and the Solution type; relax() (the
    differentiation point) exists on pdal.Solver and ipm.Solver only.
    """
    return _lookup(_BACKENDS, method).Solver()


def Settings(method="das"):
    """Factory: a default settings object for the given backend
    (elastiqp.das.Settings / pdal.Settings / ipm.Settings); assign to
    solver.settings."""
    return _lookup(_BACKENDS, method).Settings()


__all__ = [
    "METHODS",
    "das",
    "pdal",
    "ipm",
    "RowState",
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

del _core, _import_core, _name, _mod
