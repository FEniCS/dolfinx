# Copyright (C) 2024 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Utility functions for calling PETSc C functions from Numba functions."""

from __future__ import annotations

import ctypes as _ctypes
import os
import pathlib
import warnings

import numpy as np

__all__ = ["cffi_utils", "numba_utils", "ctypes_utils"]


def get_petsc_lib() -> pathlib.Path:
    """Find the full path of the PETSc shared library.

    Returns:
        Full path to the PETSc shared library.

    Raises:
        RuntimeError: If PETSc library cannot be found for if more than
            one library is found.
    """
    import petsc4py as _petsc4py

    petsc_dir = _petsc4py.get_config()["PETSC_DIR"]
    petsc_arch = _petsc4py.lib.getPathArchPETSc()[1]
    candidate_paths = [
        os.path.join(petsc_dir, petsc_arch, "lib", "libpetsc.so"),
        os.path.join(petsc_dir, petsc_arch, "lib", "libpetsc.dylib"),
    ]
    exists_paths = []
    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            exists_paths.append(candidate_path)

    if len(exists_paths) == 0:
        raise RuntimeError(
            f"Could not find a PETSc shared library. Candidate paths: {candidate_paths}"
        )
    elif len(exists_paths) > 1:
        raise RuntimeError(f"More than one PETSc shared library found. Paths: {exists_paths}")

    return pathlib.Path(exists_paths[0])


class numba_utils:
    """Utility attributes for working with Numba and PETSc.

    These attributes are convenience functions for calling PETSc C
    functions from within Numba functions.

    Note:
        `Numba <https://numba.pydata.org/>`_ must be available
        to use these utilities.

    Examples:
        A typical use of these utility functions is::

            import numpy as np
            import numpy.typing as npt
            def set_vals(A: int,
                         m: int, rows: npt.NDArray[PETSc.IntType],
                         n: int, cols: npt.NDArray[PETSc.IntType],
                         data: npt.NDArray[PETSc.ScalarTYpe], mode: int):
                MatSetValuesLocal(A, m, rows.ctypes, n, cols.ctypes, data.ctypes, mode)
    """

    try:
        import petsc4py.PETSc as _PETSc

        import llvmlite as _llvmlite
        import numba as _numba

        _llvmlite.binding.load_library_permanently(str(get_petsc_lib()))

        _int = _numba.from_dtype(_PETSc.IntType)  # type: ignore
        _scalar = _numba.from_dtype(_PETSc.ScalarType)  # type: ignore
        _real = _numba.from_dtype(_PETSc.RealType)  # type: ignore
        _int_ptr = _numba.core.types.CPointer(_int)
        _scalar_ptr = _numba.core.types.CPointer(_scalar)
        _MatSetValues_sig = _numba.core.typing.signature(
            _numba.core.types.intc,
            _numba.core.types.uintp,
            _int,
            _int_ptr,
            _int,
            _int_ptr,
            _scalar_ptr,
            _numba.core.types.intc,
        )
        MatSetValuesLocal = _numba.core.types.ExternalFunction(
            "MatSetValuesLocal", _MatSetValues_sig
        )
        """See PETSc `MatSetValuesLocal
        <https://petsc.org/release/manualpages/Mat/MatSetValuesLocal>`_
        documentation."""

        MatSetValuesBlockedLocal = _numba.core.types.ExternalFunction(
            "MatSetValuesBlockedLocal", _MatSetValues_sig
        )
        """See PETSc `MatSetValuesBlockedLocal
        <https://petsc.org/release/manualpages/Mat/MatSetValuesBlockedLocal>`_
        documentation."""
    except ImportError:
        pass


class ctypes_utils:
    """Utility attributes for working with ctypes and PETSc.

    These attributes are convenience functions for calling PETSc C
    functions, typically from within Numba functions.

    Examples:
        A typical use of these utility functions is::

            import numpy as np
            import numpy.typing as npt
            def set_vals(A: int,
                         m: int, rows: npt.NDArray[PETSc.IntType],
                         n: int, cols: npt.NDArray[PETSc.IntType],
                         data: npt.NDArray[PETSc.ScalarTYpe], mode: int):
                MatSetValuesLocal(A, m, rows.ctypes, n, cols.ctypes, data.ctypes, mode)
    """

    try:
        import petsc4py.PETSc as _PETSc

        _lib_ctypes = _ctypes.cdll.LoadLibrary(str(get_petsc_lib()))

        # Note: ctypes does not have complex types, hence we use void* for
        # scalar data
        _int = np.ctypeslib.as_ctypes_type(_PETSc.IntType)  # type: ignore

        MatSetValuesLocal = _lib_ctypes.MatSetValuesLocal
        """See PETSc `MatSetValuesLocal
        <https://petsc.org/release/manualpages/Mat/MatSetValuesLocal>`_
        documentation."""
        MatSetValuesLocal.argtypes = [
            _ctypes.c_void_p,
            _int,
            _ctypes.POINTER(_int),
            _int,
            _ctypes.POINTER(_int),
            _ctypes.c_void_p,
            _ctypes.c_int,
        ]

        MatSetValuesBlockedLocal = _lib_ctypes.MatSetValuesBlockedLocal
        """See PETSc `MatSetValuesBlockedLocal
        <https://petsc.org/release/manualpages/Mat/MatSetValuesBlockedLocal>`_
        documentation."""
        MatSetValuesBlockedLocal.argtypes = [
            _ctypes.c_void_p,
            _int,
            _ctypes.POINTER(_int),
            _int,
            _ctypes.POINTER(_int),
            _ctypes.c_void_p,
            _ctypes.c_int,
        ]
    except ImportError:
        pass


class cffi_utils:
    """Utility attributes for working with CFFI (ABI mode) and Numba.

    Registers Numba's complex types with CFFI.

    If PETSc is available, CFFI convenience functions for calling PETSc C
    functions are also created. These are typically called from within Numba
    functions.

    Note:
        `CFFI <https://cffi.readthedocs.io/>`_ and  `Numba
        <https://numba.pydata.org/>`_ must be available to use these utilities.

    Examples:
        A typical use of these utility functions is::

            import numpy as np
            import numpy.typing as npt
            def set_vals(A: int,
                         m: int, rows: npt.NDArray[PETSc.IntType],
                         n: int, cols: npt.NDArray[PETSc.IntType],
                         data: npt.NDArray[PETSc.ScalarType], mode: int):
                MatSetValuesLocal(A, m, ffi.from_buffer(rows), n, ffi.from_buffer(cols),
                                ffi.from_buffer(rows(data), mode)
    """

    import cffi as _cffi

    _ffi = _cffi.FFI()

    try:
        import numba as _numba
        import numba.core.typing.cffi_utils as _cffi_support

        # Register complex types
        _cffi_support.register_type(_ffi.typeof("float _Complex"), _numba.types.complex64)
        _cffi_support.register_type(_ffi.typeof("double _Complex"), _numba.types.complex128)

    except KeyError:
        pass
    except ImportError:
        warnings.warn("Could not import numba, so complex types are not registered!", ImportWarning)

    try:
        from petsc4py import PETSc as _PETSc

        _lib_cffi = _ffi.dlopen(str(get_petsc_lib()))

        _CTYPES = {
            np.int32: "int32_t",
            np.int64: "int64_t",
            np.float32: "float",
            np.float64: "double",
            np.complex64: "float _Complex",
            np.complex128: "double _Complex",
            np.longlong: "long long",
        }

        _c_int_t = _CTYPES[_PETSc.IntType]  # type: ignore
        _c_scalar_t = _CTYPES[_PETSc.ScalarType]  # type: ignore
        _ffi.cdef(
            f"""
                int MatSetValuesLocal(void* mat, {_c_int_t} nrow, const {_c_int_t}* irow,
                                    {_c_int_t} ncol, const {_c_int_t}* icol,
                                    const {_c_scalar_t}* y, int addv);
                int MatSetValuesBlockedLocal(void* mat, {_c_int_t} nrow, const {_c_int_t}* irow,
                                    {_c_int_t} ncol, const {_c_int_t}* icol,
                                    const {_c_scalar_t}* y, int addv);
                                    """
        )

        MatSetValuesLocal = _lib_cffi.MatSetValuesLocal
        """See PETSc `MatSetValuesLocal
        <https://petsc.org/release/manualpages/Mat/MatSetValuesLocal>`_
        documentation."""

        MatSetValuesBlockedLocal = _lib_cffi.MatSetValuesBlockedLocal
        """See PETSc `MatSetValuesBlockedLocal
        <https://petsc.org/release/manualpages/Mat/MatSetValuesBlockedLocal>`_
        documentation."""
    except KeyError:
        pass
    except ImportError:
        warnings.warn(
            "Could not import petsc4py, so numba petsc overloads are not available!", ImportWarning
        )
