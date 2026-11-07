# Copyright (C) 2026 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for PETSc vector helpers."""

from mpi4py import MPI

import numpy as np
import pytest

from dolfinx.common import index_map as create_index_map


@pytest.mark.petsc4py
def test_nest_ghost_update_duplicated_subvectors() -> None:
    """Ghost update a nest whose sub-vectors share a scatter context.

    A ghosted vector and its ``duplicate()`` share PETSc's ghost
    scatter, which allows only one operation in flight at a time.
    """
    from petsc4py import PETSc

    from dolfinx.la.petsc import _ghost_update, create_vector

    comm = MPI.COMM_WORLD
    n = 3
    owner = (comm.rank + 1) % comm.size
    ghosts = (
        np.array([owner * n] if comm.size > 1 else [], dtype=np.int64),
        np.array([owner] if comm.size > 1 else [], dtype=np.int32),
    )
    imap = create_index_map(comm, n, ghosts)
    v = create_vector([(imap, 1)])
    w = v.duplicate()
    x = PETSc.Vec().createNest([v, w], comm=comm)
    try:
        offset = comm.rank * n
        for scale, sub in enumerate((v, w), start=1):
            with sub.localForm() as local:
                local.array_w[:n] = scale * (offset + np.arange(n))
                local.array_w[n:] = -1
        _ghost_update(x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)  # type: ignore[arg-type]
        for scale, sub in enumerate((v, w), start=1):
            with sub.localForm() as local:
                np.testing.assert_array_equal(local.array_r[n:], scale * ghosts[0])
    finally:
        x.destroy()
        w.destroy()
        v.destroy()
