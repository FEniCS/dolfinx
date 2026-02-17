# Copyright (C) 2025 Garth N. Wells, Jack S. Hale
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Functions for working with PETSc linear algebra objects.

Note:
    Due to subtle issues in the interaction between petsc4py memory
    management and the Python garbage collector, it is recommended that
    the PETSc method ``destroy()`` is called on returned PETSc objects
    once the object is no longer required. Note that ``destroy()`` is
    collective over the object's MPI communicator.
"""

import functools
import itertools
import typing
from collections.abc import Iterable, Sequence

from petsc4py import PETSc

import numpy as np
import numpy.typing as npt

import dolfinx
from dolfinx.common import IndexMap
from dolfinx.la import Vector

assert dolfinx.has_petsc4py

__all__ = ["assign", "create_vector", "create_vector_wrap"]


def _ghost_update(x: PETSc.Vec, insert_mode: PETSc.InsertMode, scatter_mode: PETSc.ScatterMode):
    """Helper function for ghost updating PETSc vectors."""
    if x.getType() == PETSc.Vec.Type.NEST:
        for x_sub in x.getNestSubVecs():
            x_sub.ghostUpdate(addv=insert_mode, mode=scatter_mode)
            x_sub.destroy()
    else:
        x.ghostUpdate(addv=insert_mode, mode=scatter_mode)


def _zero_vector(x: PETSc.Vec):
    """Helper function for zeroing out PETSc vectors."""
    if x.getType() == PETSc.Vec.Type.NEST:
        for x_sub in x.getNestSubVecs():
            with x_sub.localForm() as x_sub_local:
                x_sub_local.set(0.0)
            x_sub.destroy()
    else:
        with x.localForm() as x_local:
            x_local.set(0.0)


def create_vector_wrap(x: Vector) -> PETSc.Vec:
    """Wrap a distributed DOLFINx vector as a PETSc vector.

    Args:
        x: The vector to wrap as a PETSc vector.

    Returns:
        A PETSc vector that shares data with ``x``.
    """
    index_map = x.index_map
    ghosts = index_map.ghosts.astype(PETSc.IntType)  # type: ignore[attr-defined]
    bs = x.block_size
    size = (index_map.size_local * bs, index_map.size_global * bs)

    # TODO: needs stub fix in PETSc
    return PETSc.Vec().createGhostWithArray(
        ghosts,
        x.array,  # type: ignore[arg-type]
        size=size,
        bsize=bs,
        comm=index_map.comm,
    )


def create_vector(
    maps: typing.Sequence[tuple[IndexMap, int]], kind: str | None = None
) -> PETSc.Vec:
    """Create a PETSc vector from a sequence of maps and blocksizes.

    Three cases are supported:

    1. If ``maps=[(im_0, bs_0), ..., (im_n, bs_n)]`` is a sequence of
       indexmaps and blocksizes and ``kind`` is ``None``or is
       ``PETSc.Vec.Type.MPI``, a ghosted PETSc vector whith block structure
       described by ``(im_i, bs_i)`` is created.
       The created vector ``b`` is initialized such that on each MPI
       process ``b = [b_0, b_1, ..., b_n, b_0g, b_1g, ..., b_ng]``, where
       ``b_i`` are the entries associated with the 'owned' degrees-of-
       freedom for ``(im_i, bs_i)`` and ``b_ig`` are the 'unowned' (ghost)
       entries.

       If more than one tuple is supplied, the returned vector has an
       attribute ``_blocks`` that holds the local offsets into ``b`` for
       the (i) owned and (ii) ghost entries for each ``V[i]``. It can be
       accessed by ``b.getAttr("_blocks")``. The offsets can be used to get
       views into ``b`` for blocks, e.g.::

           >>> offsets0, offsets1, = b.getAttr("_blocks")
           >>> offsets0
           (0, 12, 28)
           >>> offsets1
           (28, 32, 35)
           >>> b0_owned = b.array[offsets0[0]:offsets0[1]]
           >>> b0_ghost = b.array[offsets1[0]:offsets1[1]]
           >>> b1_owned = b.array[offsets0[1]:offsets0[2]]
           >>> b1_ghost = b.array[offsets1[1]:offsets1[2]]

    2. If ``V=[(im_0, bs_0), ..., (im_n, bs_n)]`` is a sequence of function
       space and ``kind`` is ``PETSc.Vec.Type.NEST``, a PETSc nested vector
       (a 'nest' of ghosted PETSc vectors) is created.

    Args:
        maps: Sequence of tuples of ``IndexMap`` and the associated
            block size.
        kind: PETSc vector type (``VecType``) to create.

    Returns:
        A PETSc vector with the prescribed layout. The vector is not
        initialised to zero.
    """
    if len(maps) == 1:
        # Single space case
        index_map, bs = maps[0]
        ghosts = index_map.ghosts.astype(PETSc.IntType)  # type: ignore[attr-defined]
        size = (index_map.size_local * bs, index_map.size_global * bs)
        b = PETSc.Vec().createGhost(ghosts, size=size, bsize=bs, comm=index_map.comm)
        if kind == PETSc.Vec.Type.MPI:
            _assign_block_data(maps, b)
        return b

    if kind is None or kind == PETSc.Vec.Type.MPI:
        b = dolfinx.cpp.fem.petsc.create_vector_block(maps)
        _assign_block_data(maps, b)
        return b
    elif kind == PETSc.Vec.Type.NEST:
        return dolfinx.cpp.fem.petsc.create_vector_nest(maps)
    else:
        raise NotImplementedError(
            "Vector type must be specified for blocked/nested assembly."
            f"Vector type '{kind}' not supported."
            "Did you mean 'nest' or 'mpi'?"
        )


@functools.singledispatch
def assign(
    x0: PETSc.Vec | npt.NDArray[np.inexact] | Sequence[npt.NDArray[np.inexact]],
    x1: PETSc.Vec,
):
    """Assign ``x0`` values to a PETSc vector ``x1``.

    Values in ``x0``, which is possibly a stacked collection of arrays,
    are assigned ``x1``. When ``x0`` holds a sequence of ``n``` arrays
    and ``x1`` has type ``NEST``, the assignment is::

              [x0[0]]
        x1 =  [x0[1]]
              [.....]
              [x0[n-1]]

    When ``x0`` holds a sequence of ``n`` arrays and ``x1`` **does
    not** have type ``NEST``, the assignment is::

              [x0_owned[0]]
        x1 =  [.....]
              [x0_owned[n-1]]
              [x0_ghost[0]]
              [.....]
              [x0_ghost[n-1]]

    Args:
        x0: An array or list of arrays that will be assigned to ``x1``.
        x1: Vector to assign values to.
    """
    if x1.getType() == PETSc.Vec.Type().NEST:
        x1_nest = x1.getNestSubVecs()
        for _x0, _x1 in zip(x0, x1_nest):
            with _x1.localForm() as x:
                x.array_w[:] = _x0
    else:
        with x1.localForm() as _x:
            if isinstance(x0, Sequence):
                start = 0
                for _x0 in x0:
                    end = start + _x0.shape[0]
                    _x.array_w[start:end] = _x0
                    start = end
            else:
                _x.array_w[:] = x0


@assign.register
def _(
    x0: PETSc.Vec,
    x1: npt.NDArray[np.inexact] | Sequence[npt.NDArray[np.inexact]],
):
    """Assign PETSc vector ``x0`` values to (blocked) array(s) ``x1``.

    This function performs the reverse of the assignment performed by
    the version of :func:`.assign(x0: (npt.NDArray[np.inexact] |
    list[npt.NDArray[np.inexact]]), x1: PETSc.Vec)`.

    Args:
        x0: Vector that will have its values assigned to ``x1``.
        x1: An array or list of arrays to assign to.
    """
    if x0.getType() == PETSc.Vec.Type().NEST:
        x0_nest = x0.getNestSubVecs()
        for _x0, _x1 in zip(x0_nest, x1):
            with _x0.localForm() as x:
                _x1[:] = x.array_r[:]
    else:
        with x0.localForm() as _x0:
            if isinstance(x1, Sequence):
                start = 0
                for _x1 in x1:
                    end = start + _x1.shape[0]
                    _x1[:] = _x0.array_r[start:end]
                    start = end
            else:
                x1[:] = _x0.array_r[:]


def _assign_block_data(maps: Iterable[tuple[IndexMap, int]], vec: PETSc.Vec):
    """Assign block data to a PETSc vector.

    Args:
        maps: Iterable of tuples each containing an ``IndexMap`` and the
        associated block size ``bs``.
        vec: PETSc vector to assign block data to.
    """
    # Early exit if the vector already has block data or is a nest vector
    if vec.getAttr("_blocks") is not None or vec.getType() == "nest":
        return

    maps = [(index_map, bs) for index_map, bs in maps]
    off_owned = tuple(
        itertools.accumulate(maps, lambda off, m: off + m[0].size_local * m[1], initial=0)
    )
    off_ghost = tuple(
        itertools.accumulate(
            maps, lambda off, m: off + m[0].num_ghosts * m[1], initial=off_owned[-1]
        )
    )
    vec.setAttr("_blocks", (off_owned, off_ghost))
