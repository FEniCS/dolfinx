"""Check the MPI library matches the one family the DOLFINx wheels link."""

import sys

_mpi_package = {"darwin": "openmpi", "win32": "impi-rt"}.get(sys.platform, "mpich")
try:
    from mpi4py import MPI as _MPI

    _mpi_library = _MPI.Get_library_version()
except RuntimeError:
    _mpi_library = ""
if (
    not _mpi_library
    or ("Open MPI" in _mpi_library) != (sys.platform == "darwin")
    or (sys.platform == "win32" and "Intel" not in _mpi_library)
):
    raise ImportError(f"DOLFINx wheels need a compatible MPI library: pip install {_mpi_package}")
del _MPI, _mpi_package, _mpi_library
