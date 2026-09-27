# Copyright (C) 2017-2026 Garth N. Wells, Jack S. Hale
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Linear algebra functionality."""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING, Generic, TypeVar

from mpi4py import MPI as _MPI

import numpy as np
import numpy.typing as npt

import dolfinx
from dolfinx import cpp as _cpp
from dolfinx.common import IndexMap, Scatterer
from dolfinx.cpp.la import BlockMode, InsertMode, Norm
from dolfinx.typing import Scalar

if TYPE_CHECKING:
    from petsc4py import PETSc

    from scipy import sparse as _sparse

__all__ = [
    "InsertMode",
    "MatrixCSR",
    "Norm",
    "SparsityPattern",
    "Vector",
    "is_orthonormal",
    "matrix_csr",
    "norm",
    "orthonormalize",
    "sparsity_pattern",
    "sparsity_pattern_blocked",
    "vector",
]


_T = TypeVar("_T", np.float32, np.float64, np.complex64, np.complex128, np.int8, np.int32, np.int64)


class Vector(Generic[_T]):
    """Distributed vector object."""

    _cpp_object: (
        _cpp.la.Vector_float32
        | _cpp.la.Vector_float64
        | _cpp.la.Vector_complex64
        | _cpp.la.Vector_complex128
        | _cpp.la.Vector_int8
        | _cpp.la.Vector_int32
        | _cpp.la.Vector_int64
    )

    def __init__(
        self,
        x: (
            _cpp.la.Vector_float32
            | _cpp.la.Vector_float64
            | _cpp.la.Vector_complex64
            | _cpp.la.Vector_complex128
            | _cpp.la.Vector_int8
            | _cpp.la.Vector_int32
            | _cpp.la.Vector_int64
        ),
    ):
        """Create a distributed vector.

        Args:
            x: C++ Vector object.

        Note:
            This initialiser is intended for internal library use only.
            User code should call :func:`vector` to create a vector object.
        """
        self._cpp_object = x

    def __del__(self) -> None:
        """Delete the PETSc vector if it was created."""
        if (petsc_x := self.__dict__.get("petsc_vec")) is not None:
            petsc_x.destroy()

    @functools.cached_property
    def index_map(self) -> IndexMap:
        """Index map that describes size and parallel distribution.

        Note:
            This is a cached property. The wrapper is built on first
            access and the same object is returned thereafter.
        """
        return IndexMap(self._cpp_object.index_map)

    @property
    def block_size(self) -> int:
        """Block size for the vector."""
        return self._cpp_object.bs

    @functools.cached_property
    def scatterer(self) -> Scatterer:
        """Scatterer used for ghost communication.

        Note:
            This is a cached property. The wrapper is built on first
            access and the same object is returned thereafter.
        """
        return Scatterer(self._cpp_object.scatterer)

    @property
    def array(self) -> npt.NDArray[_T]:
        """Local representation of the vector."""
        return self._cpp_object.array  # type: ignore[return-value]

    @functools.cached_property
    def petsc_vec(self) -> PETSc.Vec:
        """PETSc vector holding the entries of the vector.

        Upon first access, this creates a PETSc ``Vec`` object that
        wraps the degree-of-freedom data. The ``Vec`` object is cached
        and the cached ``Vec`` is returned on subsequent accesses.

        Note:
          When the object is destroyed it will destroy the underlying
          petsc4py vector automatically.
        """
        if not dolfinx.has_petsc4py:
            raise RuntimeError("DOLFINx has not been built with petsc4py support.")

        from dolfinx.la.petsc import create_vector_wrap

        return create_vector_wrap(self)

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


class SparsityPattern:
    """Sparsity pattern of a distributed sparse matrix.

    A pattern is built by inserting (row, column) index pairs and then
    finalizing. Once finalized, it defines the nonzero structure and the
    parallel distribution of a :class:`MatrixCSR`.
    """

    _cpp_object: _cpp.la.SparsityPattern

    def __init__(self, sp: _cpp.la.SparsityPattern):
        """Create a sparsity pattern.

        Note:
            Objects of this type should be created using
            :func:`sparsity_pattern`, :func:`sparsity_pattern_blocked`
            or :func:`dolfinx.fem.create_sparsity_pattern`, and not
            using this initialiser.

        Args:
            sp: The C++/nanobind sparsity pattern object.
        """
        self._cpp_object = sp

    def index_map(self, dim: int) -> IndexMap:
        """Index map for the rows (``dim=0``) or columns (``dim=1``).

        Args:
            dim: 0 for the row map, 1 for the column map.
        """
        return IndexMap(self._cpp_object.index_map(dim))

    @property
    def num_nonzeros(self) -> int:
        """Number of nonzeros in the finalized pattern."""
        return self._cpp_object.num_nonzeros

    def insert(
        self,
        rows: int | npt.NDArray[np.int32],
        cols: int | npt.NDArray[np.int32],
    ) -> None:
        """Insert entries into the pattern.

        Given arrays of rows and columns, an entry is inserted for every
        (row, column) pair. Given a single row and column, the one entry
        is inserted.

        Args:
            rows: Row index/indices.
            cols: Column index/indices.
        """
        self._cpp_object.insert(rows, cols)  # type: ignore[arg-type]

    def insert_diagonal(self, rows: npt.NDArray[np.int32]) -> None:
        """Insert the diagonal entry for each of ``rows``.

        Args:
            rows: Rows to insert the diagonal entry for.
        """
        self._cpp_object.insert_diagonal(rows)

    def finalize(self) -> None:
        """Finalize the pattern.

        The pattern cannot be modified after finalizing, and it must be
        finalized before a matrix can be created from it.
        """
        self._cpp_object.finalize()

    @property
    def graph(self) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int64]]:
        """Finalized pattern as a (column indices, row offsets) pair.

        Note:
            The returned arrays are read-only views into the pattern.
        """
        return self._cpp_object.graph


class MatrixCSR(Generic[Scalar]):
    """Distributed compressed sparse row matrix."""

    _cpp_object: (
        _cpp.la.MatrixCSR_float32
        | _cpp.la.MatrixCSR_float64
        | _cpp.la.MatrixCSR_complex64
        | _cpp.la.MatrixCSR_complex128
    )

    def __init__(
        self,
        A: (
            _cpp.la.MatrixCSR_float32
            | _cpp.la.MatrixCSR_float64
            | _cpp.la.MatrixCSR_complex64
            | _cpp.la.MatrixCSR_complex128
        ),
    ):
        """Create a distributed compressed sparse row matrix.

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
        return IndexMap(self._cpp_object.index_map(i))

    def mult(self, x: Vector[Scalar], y: Vector[Scalar], transpose: bool = False) -> None:
        """Compute ``y += Ax`` or ``y += A^T x``.

        Args:
            x: Input Vector
            y: Output Vector
            transpose: if True, compute y += A^T x
        """
        if transpose:
            self._cpp_object.multT(x._cpp_object, y._cpp_object)  # type: ignore[arg-type]
        else:
            self._cpp_object.mult(x._cpp_object, y._cpp_object)  # type: ignore[arg-type]

    def matmul(self, B: MatrixCSR[Scalar]) -> MatrixCSR[Scalar]:
        """Compute matrix product ``A * B``, where `A` is this matrix.

        Args:
            B: Input Matrix to multiply by
        """
        if (
            self.index_map(1).size_local != B.index_map(0).size_local
            or self.index_map(1).size_global != B.index_map(0).size_global
        ):
            raise RuntimeError("Invalid matrix sizes for matmul.")
        if (
            self.block_size[0] != 1
            or self.block_size[1] != 1
            or B.block_size[0] != 1
            or B.block_size[1] != 1
        ):
            raise RuntimeError("Block size not supported in matmul.")

        return MatrixCSR(self._cpp_object.mult(B._cpp_object))  # type: ignore[arg-type]

    def transpose(self) -> MatrixCSR[Scalar]:
        """Compute transpose matrix."""
        return MatrixCSR(self._cpp_object.transpose())

    @property
    def block_size(self) -> list:
        """Block sizes for the matrix."""
        return self._cpp_object.bs

    def add(
        self,
        x: npt.NDArray[Scalar],
        rows: npt.NDArray[np.int32],
        cols: npt.NDArray[np.int32],
        bs: int = 1,
    ) -> None:
        """Add a block of values in the matrix."""
        self._cpp_object.add(x, rows, cols, bs)  # type: ignore[arg-type]

    def set(
        self,
        x: npt.NDArray[Scalar],
        rows: npt.NDArray[np.int32],
        cols: npt.NDArray[np.int32],
        bs: int = 1,
    ) -> None:
        """Set a block of values in the matrix."""
        self._cpp_object.set(x, rows, cols, bs)  # type: ignore[arg-type]

    def set_value(self, x: Scalar) -> None:
        """Set all non-zero entries to a value.

        Args:
            x: The value to set all non-zero entries to.
        """
        self.data[:] = x

    def scatter_reverse(self) -> None:
        """Scatter and accumulate ghost values."""
        self._cpp_object.scatter_reverse()

    def eliminate_zeros(self, tol: float = 0) -> None:
        """Remove explicitly-stored entries that are within a tolerance.

        This compacts the underlying storage: entries with
        ``abs(value) <= tol`` are dropped, and the column indices and
        row pointers are updated accordingly. Entries with
        ``abs(value) > tol`` are left untouched.

        Note:
            This is a terminal, finalizing operation. It can reduce the
            matrix's sparsity, which invalidates the precomputed
            communication pattern used to accumulate ghost row
            contributions. After calling this, the matrix can no longer
            be modified: further calls to :meth:`add`, :meth:`set`, or
            :meth:`scatter_reverse` will raise a ``RuntimeError``. Only
            call this once, after the matrix is fully assembled (i.e.
            after the final :meth:`scatter_reverse`).

        Args:
            tol: Entries with magnitude less than or equal to ``tol``
                are removed from storage. Defaults to removing only
                exact zeros.
        """
        self._cpp_object.eliminate_zeros(self.data.dtype.type(tol))  # type: ignore[arg-type]

    def squared_norm(self) -> float:
        """Compute the squared Frobenius norm.

        Note:
            This operation is collective and requires communication.
        """
        return self._cpp_object.squared_norm()

    @property
    def data(self) -> npt.NDArray[Scalar]:
        """Underlying matrix entry data."""
        return self._cpp_object.data  # type: ignore[return-value]

    @property
    def indices(self) -> npt.NDArray[np.int32]:
        """Local column indices."""
        return self._cpp_object.indices

    @property
    def indptr(self) -> npt.NDArray[np.int64]:
        """Local row pointers."""
        return self._cpp_object.indptr

    def to_dense(self) -> npt.NDArray[Scalar]:
        """Copy to a dense 2D array.

        Note:
            Typically used for debugging.
        """
        return self._cpp_object.to_dense()  # type: ignore[return-value]

    def to_scipy(self, ghosted: bool = False) -> _sparse.csr_matrix | _sparse.bsr_matrix:
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


def sparsity_pattern(
    comm: _MPI.Comm, maps: Sequence[IndexMap], bs: Sequence[int]
) -> SparsityPattern:
    """Create a sparsity pattern for a matrix.

    Args:
        comm: MPI communicator that the pattern is distributed over.
        maps: Row and column index maps.
        bs: Row and column block sizes.

    Returns:
        An empty sparsity pattern. Insert entries into it and call
        :meth:`SparsityPattern.finalize` before creating a matrix.
    """
    return SparsityPattern(_cpp.la.SparsityPattern(comm, [m._cpp_object for m in maps], list(bs)))


def sparsity_pattern_blocked(
    comm: _MPI.Comm,
    patterns: Sequence[Sequence[SparsityPattern | None]],
    maps: Sequence[Sequence[tuple[IndexMap, int]]],
    bs: Sequence[Sequence[int]],
) -> SparsityPattern:
    """Create a sparsity pattern from a rectangular array of patterns.

    The blocks are concatenated into a single pattern, as required for a
    monolithic matrix assembled from a block form.

    Args:
        comm: MPI communicator that the pattern is distributed over.
        patterns: Sparsity pattern of each block. ``None`` marks a
            structurally zero block.
        maps: Index map and block size of each block row, and of each
            block column.
        bs: Row and column block sizes of the assembled pattern.

    Returns:
        An unfinalized sparsity pattern spanning all blocks.
    """
    return SparsityPattern(
        _cpp.la.SparsityPattern(
            comm,
            # The C++ constructor permits null blocks, but the generated
            # stub renders the nested pointer as non-optional.
            [[p._cpp_object if p is not None else None for p in row] for row in patterns],  # type: ignore[misc]
            [[(m._cpp_object, mbs) for m, mbs in row] for row in maps],
            [list(b) for b in bs],
        )
    )


def matrix_csr(
    sp: SparsityPattern,
    block_mode: BlockMode = BlockMode.compact,
    dtype: npt.DTypeLike = np.float64,
) -> MatrixCSR:
    """Create a distributed sparse matrix.

    The matrix uses compressed sparse row storage.

    Args:
        sp: The sparsity pattern that defines the nonzero structure of
            the matrix the parallel distribution of the matrix.
        block_mode: Block mode to use.
        dtype: Scalar type.

    Returns:
        A sparse matrix.
    """
    ftype: (
        type[_cpp.la.MatrixCSR_float32]
        | type[_cpp.la.MatrixCSR_float64]
        | type[_cpp.la.MatrixCSR_complex64]
        | type[_cpp.la.MatrixCSR_complex128]
    )
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

    return MatrixCSR(ftype(sp._cpp_object, block_mode))


def vector(
    map: IndexMap,
    bs: int = 1,
    scatterer: Scatterer | None = None,
    *,
    dtype: npt.DTypeLike = np.float64,
) -> Vector:
    """Create a distributed vector.

    Args:
        map: Index map the describes the size and distribution of the
            vector.
        bs: Block size.
        scatterer: Scatterer compatible with ``map``. If ``None``, a
            new scatterer is created.
        dtype: The scalar type.

    Returns:
        A distributed vector.
    """
    vtype: (
        type[_cpp.la.Vector_float32]
        | type[_cpp.la.Vector_float64]
        | type[_cpp.la.Vector_complex64]
        | type[_cpp.la.Vector_complex128]
        | type[_cpp.la.Vector_int8]
        | type[_cpp.la.Vector_int32]
        | type[_cpp.la.Vector_int64]
    )
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

    if scatterer is None:
        return Vector(vtype(map._cpp_object, bs))
    else:
        return Vector(vtype(map._cpp_object, bs, scatterer._cpp_object))


def orthonormalize(basis: list[Vector[_T]]) -> None:
    """Orthogonalise set of vectors in-place."""
    _cpp.la.orthonormalize([x._cpp_object for x in basis])  # type: ignore[misc]


def is_orthonormal(basis: list[Vector[_T]], eps: float = 1.0e-12) -> bool:
    """Check that list of vectors are orthonormal."""
    return _cpp.la.is_orthonormal([x._cpp_object for x in basis], eps)  # type: ignore[misc]


def norm(x: Vector[_T], type: _cpp.la.Norm = _cpp.la.Norm.l2) -> float:
    """Compute a norm of the vector.

    Args:
        x: Vector to measure.
        type: Norm type to compute.

    Returns:
        Computed norm.
    """
    return _cpp.la.norm(x._cpp_object, type)  # type: ignore[arg-type]
