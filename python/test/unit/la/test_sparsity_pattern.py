# Copyright (C) 2017 Nathan Sime
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for SparsityPattern"""

import numpy as np
import pytest

from dolfin import MPI, CellType, FunctionSpace, UnitSquareMesh, cpp
from dolfin_utils.test.fixtures import fixture


def count_on_and_off_diagonal_nnz(primary_codim_entries, local_range):
    nnz_on_diagonal = sum(1 for entry in primary_codim_entries
                          if local_range[0] <= entry < local_range[1])
    nnz_off_diagonal = sum(1 for entry in primary_codim_entries
                           if not local_range[0] <= entry < local_range[1])
    return nnz_on_diagonal, nnz_off_diagonal


@fixture
def mesh():
    return UnitSquareMesh(MPI.comm_world, 4, 4, CellType.Type.triangle)


@fixture
def V(mesh):
    return FunctionSpace(mesh, ("Lagrange", 1))


def xtest_str(mesh, V):
    dm = V.dofmap()
    index_map = dm.index_map()
    assert index_map

    # Build sparse tensor layout (for assembly of matrix)
    sp = cpp.fem.SparsityPatternBuilder.build(mesh.mpi_comm(), mesh,
                                              [dm._cpp_object, dm._cpp_object],
                                              True, False, False)
    sp.assemble()

    sp.str(False)
    sp.str(True)


@pytest.mark.xfail
def test_insert_local(mesh, V):
    dm = V.dofmap()
    index_map = dm.index_map()
    assert index_map

    sp = cpp.fem.SparsityPatternBuilder.build(mesh.mpi_comm(), mesh,
                                              [dm._cpp_object, dm._cpp_object],
                                              True, False, False)
    sp.assemble()

    sp1 = cpp.la.SparsityPattern(mesh.mpi_comm(), [[sp], [sp]])
    if (MPI.rank(mesh.mpi_comm()) == 0):
        print("\nPattern:")
        print(sp1.str(True))

    sp1 = cpp.la.SparsityPattern(mesh.mpi_comm(), [[sp, sp]])
    if (MPI.rank(mesh.mpi_comm()) == 0):
        print("\nPattern:")
        print(sp1.str(True))

    sp1 = cpp.la.SparsityPattern(mesh.mpi_comm(), [[sp, sp], [sp, sp]])
    if (MPI.rank(mesh.mpi_comm()) == 0):
        print("\nPattern:")
        print(sp1.str(True))


def xtest_insert_global(mesh, V):
    dm = V.dofmap()
    index_map = dm.index_map()
    local_range = index_map.local_range()

    # Build sparse tensor layout
    tl = cpp.la.TensorLayout(mesh.mpi_comm(), 0,
                             cpp.la.TensorLayout.Sparsity.SPARSE)
    tl.init([index_map, index_map], cpp.la.TensorLayout.Ghosts.UNGHOSTED)
    sp = tl.sparsity_pattern()
    sp.init([index_map, index_map])

    # Primary dim (row) entries need to be local to the process, so we ensure
    # they're in the local range of the index map
    primary_dim_local_entries = np.array([0, 1, 2], dtype=np.intc)
    primary_dim_entries = primary_dim_local_entries + local_range[0]

    # The codim (column) entries will be added to the same global entries
    # on each process.
    primary_codim_entries = np.array([0, 1, 2], dtype=np.intc)
    entries = np.array([primary_dim_entries, primary_codim_entries],
                       dtype=np.intc)

    sp.insert_global(entries)
    sp.apply()

    nnz_d = sp.num_nonzeros_diagonal()
    nnz_od = sp.num_nonzeros_off_diagonal()

    # rank = MPI.rank(mesh.mpi_comm())
    # size = MPI.size(mesh.mpi_comm())

    # Tabulate on diagonal and off diagonal nnzs
    nnz_on_diagonal, nnz_off_diagonal = count_on_and_off_diagonal_nnz(
        primary_codim_entries, local_range)

    # Compare tabulated and sparsity pattern nnzs
    for local_row in range(len(nnz_d)):
        if local_range[0] <= local_row < local_range[1]:
            assert nnz_d[local_row] == (nnz_on_diagonal if
                                        local_row in primary_dim_local_entries
                                        else 0)
        else:
            assert nnz_od[local_row] == (nnz_off_diagonal if
                                         local_row in primary_dim_local_entries
                                         else 0)


def xtest_insert_local_global(mesh, V):
    dm = V.dofmap()
    index_map = dm.index_map()
    local_range = index_map.local_range()

    # Build sparse tensor layout
    tl = cpp.la.TensorLayout(mesh.mpi_comm(), 0,
                             cpp.la.TensorLayout.Sparsity.SPARSE)
    tl.init([index_map, index_map], cpp.la.TensorLayout.Ghosts.UNGHOSTED)
    sp = tl.sparsity_pattern()
    sp.init([index_map, index_map])

    # Primary dim (row) entries need to be local to the process, so we ensure
    # they're in the local range of the index map
    primary_dim_local_entries = np.array([0, 1, 2], dtype=np.intc)
    primary_dim_entries = primary_dim_local_entries

    # The codim (column) entries will be added to the same global entries
    # on each process.
    primary_codim_entries = np.array([0, 1, 2], dtype=np.intc)
    entries = np.array([primary_dim_entries, primary_codim_entries],
                       dtype=np.intc)

    sp.insert_local_global(entries)
    sp.apply()

    nnz_d = sp.num_nonzeros_diagonal()
    nnz_od = sp.num_nonzeros_off_diagonal()

    # rank = MPI.rank(mesh.mpi_comm())
    # size = MPI.size(mesh.mpi_comm())

    # Tabulate on diagonal and off diagonal nnzs
    nnz_on_diagonal, nnz_off_diagonal = count_on_and_off_diagonal_nnz(
        primary_codim_entries, local_range)

    # Compare tabulated and sparsity pattern nnzs
    for local_row in range(len(nnz_d)):
        if local_range[0] <= local_row < local_range[1]:
            assert nnz_d[local_row] == (nnz_on_diagonal if
                                        local_row in primary_dim_local_entries
                                        else 0)
        else:
            assert nnz_od[local_row] == (nnz_off_diagonal if
                                         local_row in primary_dim_local_entries
                                         else 0)
