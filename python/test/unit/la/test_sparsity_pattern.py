"Unit tests for SparsityPattern"

# Copyright (C) 2017 Nathan Sime
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

import pytest
import numpy as np
from dolfin import *
from dolfin_utils.test import *


def count_on_and_off_diagonal_nnz(primary_codim_entries, local_range):
    nnz_on_diagonal = sum(1 for entry in primary_codim_entries
                          if local_range[0] <= entry < local_range[1])
    nnz_off_diagonal = sum(1 for entry in primary_codim_entries
                           if not local_range[0] <= entry < local_range[1])
    return nnz_on_diagonal, nnz_off_diagonal


@fixture
def mesh():
    return UnitSquareMesh(10, 10)


@fixture
def V(mesh):
    return FunctionSpace(mesh, "CG", 1)


def test_str(mesh, V):
    dm = V.dofmap()
    index_map = dm.index_map()

    # Build sparse tensor layout (for assembly of matrix)
    tl = TensorLayout(mesh.mpi_comm(), 0, TensorLayout.Sparsity.SPARSE)
    tl.init([index_map, index_map], TensorLayout.Ghosts.UNGHOSTED)
    sp = tl.sparsity_pattern()
    sp.init([index_map, index_map])
    SparsityPatternBuilder.build(sp, mesh, [dm, dm],
                                 True, False, False, False,
                                 False, init=False, finalize=True)

    sp.str(False)
    sp.str(True)


def test_insert_local(mesh, V):
    dm = V.dofmap()
    index_map = dm.index_map()

    # Build sparse tensor layout
    tl = TensorLayout(mesh.mpi_comm(), 0, TensorLayout.Sparsity.SPARSE)
    tl.init([index_map, index_map], TensorLayout.Ghosts.UNGHOSTED)
    sp = tl.sparsity_pattern()
    sp.init([index_map, index_map])

    primary_dim_entries = [0, 1, 2]
    primary_codim_entries = [0, 1, 2]
    entries = np.array([primary_dim_entries, primary_codim_entries], dtype=np.intc)
    sp.insert_local(entries)

    sp.apply()

    assert len(primary_dim_entries) * len(primary_codim_entries) == sp.num_nonzeros()

    nnz_d = sp.num_nonzeros_diagonal()
    for local_row in range(len(nnz_d)):
        assert nnz_d[local_row] == (len(primary_codim_entries) if local_row in primary_dim_entries else 0)


def test_insert_global(mesh, V):
    dm = V.dofmap()
    index_map = dm.index_map()
    local_range = index_map.local_range()

    # Build sparse tensor layout
    tl = TensorLayout(mesh.mpi_comm(), 0, TensorLayout.Sparsity.SPARSE)
    tl.init([index_map, index_map], TensorLayout.Ghosts.UNGHOSTED)
    sp = tl.sparsity_pattern()
    sp.init([index_map, index_map])

    # Primary dim (row) entries need to be local to the process, so we ensure
    # they're in the local range of the index map
    primary_dim_local_entries = np.array([0, 1, 2], dtype=np.intc)
    primary_dim_entries = primary_dim_local_entries + local_range[0]

    # The codim (column) entries will be added to the same global entries
    # on each process.
    primary_codim_entries = np.array([0, 1, 2], dtype=np.intc)
    entries = np.array([primary_dim_entries, primary_codim_entries], dtype=np.intc)

    sp.insert_global(entries)
    sp.apply()

    nnz_d = sp.num_nonzeros_diagonal()
    nnz_od = sp.num_nonzeros_off_diagonal()

    rank = MPI.rank(mesh.mpi_comm())
    size = MPI.size(mesh.mpi_comm())

    # Tabulate on diagonal and off diagonal nnzs
    nnz_on_diagonal, nnz_off_diagonal = count_on_and_off_diagonal_nnz(
        primary_codim_entries, local_range)

    # Compare tabulated and sparsity pattern nnzs
    for local_row in range(len(nnz_d)):
        if local_range[0] <= local_row < local_range[1]:
            assert nnz_d[local_row] == (nnz_on_diagonal if local_row in primary_dim_local_entries else 0)
        else:
            assert nnz_od[local_row] == (nnz_off_diagonal if local_row in primary_dim_local_entries else 0)


def test_insert_local_global(mesh, V):
    dm = V.dofmap()
    index_map = dm.index_map()
    local_range = index_map.local_range()

    # Build sparse tensor layout
    tl = TensorLayout(mesh.mpi_comm(), 0, TensorLayout.Sparsity.SPARSE)
    tl.init([index_map, index_map], TensorLayout.Ghosts.UNGHOSTED)
    sp = tl.sparsity_pattern()
    sp.init([index_map, index_map])

    # Primary dim (row) entries need to be local to the process, so we ensure
    # they're in the local range of the index map
    primary_dim_local_entries = np.array([0, 1, 2], dtype=np.intc)
    primary_dim_entries = primary_dim_local_entries

    # The codim (column) entries will be added to the same global entries
    # on each process.
    primary_codim_entries = np.array([0, 1, 2], dtype=np.intc)
    entries = np.array([primary_dim_entries, primary_codim_entries], dtype=np.intc)

    sp.insert_local_global(entries)
    sp.apply()

    nnz_d = sp.num_nonzeros_diagonal()
    nnz_od = sp.num_nonzeros_off_diagonal()

    rank = MPI.rank(mesh.mpi_comm())
    size = MPI.size(mesh.mpi_comm())

    # Tabulate on diagonal and off diagonal nnzs
    nnz_on_diagonal, nnz_off_diagonal = count_on_and_off_diagonal_nnz(
        primary_codim_entries, local_range)

    # Compare tabulated and sparsity pattern nnzs
    for local_row in range(len(nnz_d)):
        if local_range[0] <= local_row < local_range[1]:
            assert nnz_d[local_row] == (nnz_on_diagonal if local_row in primary_dim_local_entries else 0)
        else:
            assert nnz_od[local_row] == (nnz_off_diagonal if local_row in primary_dim_local_entries else 0)
