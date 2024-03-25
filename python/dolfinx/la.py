# Copyright (C) 2017-2021 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Linear algebra functionality"""

import typing

import numpy as np
import numpy.typing as npt

from dolfinx import cpp as _cpp
from dolfinx.cpp.common import IndexMap
from dolfinx.cpp.la import BlockMode, InsertMode, Norm

__all__ = [
    "orthonormalize",
    "is_orthonormal",
    "matrix_csr",
    "vector",
    "MatrixCSR",
    "Norm",
    "InsertMode",
    "Vector",
    "create_petsc_vector",
]


class MatrixCSR:
    _cpp_object: typing.Union[
        _cpp.la.MatrixCSR_float32,
        _cpp.la.MatrixCSR_float64,
        _cpp.la.MatrixCSR_complex64,
        _cpp.la.MatrixCSR_complex128,
    ]

    def __init__(
        self,
        A: typing.Union[
            _cpp.la.MatrixCSR_float32,
            _cpp.la.MatrixCSR_float64,
            _cpp.la.MatrixCSR_complex64,
            _cpp.la.MatrixCSR_complex128,
        ],
    ):
        """A distributed sparse matrix that uses compressed sparse row storage.

        Note:
            Objects of this type should be created using
            :func:`matrix_csr` and not created using this initialiser.

        Args:
            A: The C++/nanobind matrix object.
        """
        self._cpp_object = A

    def index_map(self, i: int) -> IndexMap:
        """Index map for row/column.

        Args:
            i: 0 for row map, 1 for column map.
        """
        return self._cpp_object.index_map(i)

    @property
    def block_size(self):
        """Block sizes for the matrix."""
        return self._cpp_object.bs

    def add(
        self,
        x: npt.NDArray[np.floating],
        rows: npt.NDArray[np.int32],
        cols: npt.NDArray[np.int32],
        bs: int = 1,
    ) -> None:
        """Add a block of values in the matrix."""
        self._cpp_object.add(x, rows, cols, bs)

    def set(
        self,
        x: npt.NDArray[np.floating],
        rows: npt.NDArray[np.int32],
        cols: npt.NDArray[np.int32],
        bs: int = 1,
    ) -> None:
        """Set a block of values in the matrix."""
        self._cpp_object.set(x, rows, cols, bs)

    def set_value(self, x: np.floating) -> None:
        """Set all non-zero entries to a value.

        Args:
            x: The value to set all non-zero entries to.
        """
        self._cpp_object.set_value(x)

    def scatter_reverse(self) -> None:
        """Scatter and accumulate ghost values."""
        self._cpp_object.scatter_reverse()

    def squared_norm(self) -> np.floating:
        """Compute the squared Frobenius norm.

        Note:
            This operation is collective and requires communication.
        """
        return self._cpp_object.squared_norm()

    @property
    def data(self) -> npt.NDArray[np.floating]:
        """Underlying matrix entry data."""
        return self._cpp_object.data

    @property
    def indices(self) -> npt.NDArray[np.int32]:
        """Local column indices."""
        return self._cpp_object.indices

    @property
    def indptr(self) -> npt.NDArray[np.int64]:
        """Local row pointers."""
        return self._cpp_object.indptr

    def to_dense(self) -> npt.NDArray[np.floating]:
        """Copy to a dense 2D array.

        Note:
            Typically used for debugging.

        """
        return self._cpp_object.to_dense()

    def to_scipy(self, ghosted=False):
        """Convert to a SciPy CSR/BSR matrix. Data is shared.

        Note:
            SciPy must be available.

        Args:
            ghosted: If ``True`` rows that are ghosted in parallel are
                included in the returned SciPy matrix, otherwise ghost
                rows are not included.

        Returns:
            SciPy compressed sparse row (both block sizes equal to one)
            or a SciPy block compressed sparse row matrix.
        """
        bs0, bs1 = self._cpp_object.bs
        ncols = self.index_map(1).size_local + self.index_map(1).num_ghosts
        if ghosted:
            nrows = self.index_map(0).size_local + self.index_map(0).num_ghosts
            data, indices, indptr = self.data, self.indices, self.indptr
        else:
            nrows = self.index_map(0).size_local
            nnzlocal = self.indptr[nrows]
            data, indices, indptr = (
                self.data[: (bs0 * bs1) * nnzlocal],
                self.indices[:nnzlocal],
                self.indptr[: nrows + 1],
            )

        if bs0 == 1 and bs1 == 1:
            from scipy.sparse import csr_matrix as _csr

            return _csr((data, indices, indptr), shape=(nrows, ncols))
        else:
            from scipy.sparse import bsr_matrix as _bsr

            return _bsr(
                (data.reshape(-1, bs0, bs1), indices, indptr), shape=(bs0 * nrows, bs1 * ncols)
            )


def matrix_csr(
    sp: _cpp.la.SparsityPattern, block_mode=BlockMode.compact, dtype: npt.DTypeLike = np.float64
) -> MatrixCSR:
    """Create a distributed sparse matrix.

    The matrix uses compressed sparse row storage.

    Args:
        sp: The sparsity pattern that defines the nonzero structure of
            the matrix the parallel distribution of the matrix.
        dtype: The scalar type.

    Returns:
        A sparse matrix.
    """
    if np.issubdtype(dtype, np.float32):
        ftype = _cpp.la.MatrixCSR_float32
    elif np.issubdtype(dtype, np.float64):
        ftype = _cpp.la.MatrixCSR_float64
    elif np.issubdtype(dtype, np.complex64):
        ftype = _cpp.la.MatrixCSR_complex64
    elif np.issubdtype(dtype, np.complex128):
        ftype = _cpp.la.MatrixCSR_complex128
    else:
        raise NotImplementedError(f"Type {dtype} not supported.")

    return MatrixCSR(ftype(sp, block_mode))


class Vector:
    _cpp_object: typing.Union[
        _cpp.la.Vector_float32,
        _cpp.la.Vector_float64,
        _cpp.la.Vector_complex64,
        _cpp.la.Vector_complex128,
        _cpp.la.Vector_int8,
        _cpp.la.Vector_int32,
        _cpp.la.Vector_int64,
    ]

    def __init__(
        self,
        x: typing.Union[
            _cpp.la.Vector_float32,
            _cpp.la.Vector_float64,
            _cpp.la.Vector_complex64,
            _cpp.la.Vector_complex128,
            _cpp.la.Vector_int8,
            _cpp.la.Vector_int32,
            _cpp.la.Vector_int64,
        ],
    ):
        """A distributed vector object.

        Args:
            x: C++ Vector object.

        Note:
            This initialiser is intended for internal library use only.
            User code should call :func:`vector` to create a vector object.
        """
        self._cpp_object = x

    @property
    def index_map(self) -> IndexMap:
        """Index map that describes size and parallel distribution."""
        return self._cpp_object.index_map

    @property
    def block_size(self) -> int:
        """Block size for the vector."""
        return self._cpp_object.bs

    @property
    def array(self) -> np.ndarray:
        """Local representation of the vector."""
        return self._cpp_object.array

    def scatter_forward(self) -> None:
        """Update ghost entries."""
        self._cpp_object.scatter_forward()

    def scatter_reverse(self, mode: InsertMode) -> None:
        """Scatter ghost entries to owner.

        Args:
            mode: Control how scattered values are set/accumulated by
                owner.
        """
        self._cpp_object.scatter_reverse(mode)


def vector(map, bs=1, dtype: npt.DTypeLike = np.float64) -> Vector:
    """Create a distributed vector.

    Args:
        map: Index map the describes the size and distribution of the
            vector.
        bs: Block size.
        dtype: The scalar type.

    Returns:
        A distributed vector.
    """
    if np.issubdtype(dtype, np.float32):
        vtype = _cpp.la.Vector_float32
    elif np.issubdtype(dtype, np.float64):
        vtype = _cpp.la.Vector_float64
    elif np.issubdtype(dtype, np.complex64):
        vtype = _cpp.la.Vector_complex64
    elif np.issubdtype(dtype, np.complex128):
        vtype = _cpp.la.Vector_complex128
    elif np.issubdtype(dtype, np.int8):
        vtype = _cpp.la.Vector_int8
    elif np.issubdtype(dtype, np.int32):
        vtype = _cpp.la.Vector_int32
    elif np.issubdtype(dtype, np.int64):
        vtype = _cpp.la.Vector_int64
    else:
        raise NotImplementedError(f"Type {dtype} not supported.")

    return Vector(vtype(map, bs))


def create_petsc_vector_wrap(x: Vector):
    """Wrap a distributed DOLFINx vector as a PETSc vector.

    Args:
        x: The vector to wrap as a PETSc vector.

    Returns:
        A PETSc vector that shares data with ``x``.

    Note:
        The vector ``x`` must not be destroyed before the returned PETSc
        object.
    """
    from petsc4py import PETSc

    map = x.index_map
    ghosts = map.ghosts.astype(PETSc.IntType)  # type: ignore
    bs = x.block_size
    size = (map.size_local * bs, map.size_global * bs)
    return PETSc.Vec().createGhostWithArray(ghosts, x.array, size=size, bsize=bs, comm=map.comm)  # type: ignore


def create_petsc_vector(map, bs: int):
    """Create a distributed PETSc vector.

    Args:
        map: Index map that describes the size and parallel layout of
            the vector to create.
        bs: Block size of the vector.

    Returns:
        PETSc Vec object.
    """
    from petsc4py import PETSc

    ghosts = map.ghosts.astype(PETSc.IntType)  # type: ignore
    size = (map.size_local * bs, map.size_global * bs)
    return PETSc.Vec().createGhost(ghosts, size=size, bsize=bs, comm=map.comm)  # type: ignore


def orthonormalize(basis):
    """Orthogonalise set of PETSc vectors in-place."""
    for i, x in enumerate(basis):
        for y in basis[:i]:
            alpha = x.dot(y)
            x.axpy(-alpha, y)
        x.normalize()


def is_orthonormal(basis, eps: float = 1.0e-12) -> bool:
    """Check that list of PETSc vectors are orthonormal."""
    for x in basis:
        if abs(x.norm() - 1.0) > eps:
            return False
    for i, x in enumerate(basis[:-1]):
        for y in basis[i + 1 :]:
            if abs(x.dot(y)) > eps:
                return False
    return True


def norm(x: Vector, type: _cpp.la.Norm = _cpp.la.Norm.l2) -> np.floating:
    """Compute a norm of the vector.

    Args:
        x: Vector to measure.
        type: Norm type to compute.

    Returns:
        Computed norm.
    """
    return _cpp.la.norm(x._cpp_object, type)
