#!/usr/bin/env py.test

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

from dolfin import *
import pytest
from dolfin_utils.test import *
import numpy as np


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
    tl = TensorLayout(mesh.mpi_comm(), 0, TensorLayout.Sparsity_SPARSE)
    tl.init([index_map, index_map], TensorLayout.Ghosts_UNGHOSTED)
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

    # Build sparse tensor layout (for assembly of matrix)
    tl = TensorLayout(mesh.mpi_comm(), 0, TensorLayout.Sparsity_SPARSE)
    tl.init([index_map, index_map], TensorLayout.Ghosts_UNGHOSTED)
    sp = tl.sparsity_pattern()
    sp.init([index_map, index_map])

    pridim_entries = [0, 1, 2]
    codim_entries = [0, 1, 2]
    entries = np.array([pridim_entries, codim_entries], dtype=np.intc)
    sp.insert_local(entries)

    sp.apply()

    assert len(pridim_entries)*len(codim_entries) == sp.num_nonzeros()

    nnz_d = sp.num_nonzeros_diagonal()
    for local_row in range(len(nnz_d)):
      if local_row in pridim_entries:
        assert nnz_d[local_row] == len(codim_entries)
      else:
        assert nnz_d[local_row] == 0


def test_insert_global(mesh, V):
    dm = V.dofmap()
    index_map = dm.index_map()
    local_range = index_map.local_range()

    # Build sparse tensor layout (for assembly of matrix)
    tl = TensorLayout(mesh.mpi_comm(), 0, TensorLayout.Sparsity_SPARSE)
    tl.init([index_map, index_map], TensorLayout.Ghosts_UNGHOSTED)
    sp = tl.sparsity_pattern()
    sp.init([index_map, index_map])

    pridim_local_entries = np.array([0, 1, 2], dtype=np.intc)
    pridim_entries = pridim_local_entries + local_range[0]
    codim_entries = np.array([0, 1, 2], dtype=np.intc)
    entries = np.array([pridim_entries, codim_entries], dtype=np.intc)

    sp.insert_global(entries)
    sp.apply()

    nnz_d = sp.num_nonzeros_diagonal()
    nnz_od = sp.num_nonzeros_off_diagonal()

    rank = MPI.rank(mesh.mpi_comm())
    size = MPI.size(mesh.mpi_comm())

    # Tabulate on diangonal and off diagonal nnzs
    nnz_on_diagonal = 0
    nnz_off_diagonal = 0
    for entry in codim_entries:
      in_range = local_range[0] <= entry < local_range[1]
      if in_range:
        nnz_on_diagonal += 1
      else:
        nnz_off_diagonal += 1

    # Compare tabulated and sparsity pattern nnzs
    for local_row in range(len(nnz_d)):
      in_range = local_range[0] <= local_row < local_range[1]

      if local_row in pridim_local_entries:
        # We added some entries into this row DoF, so check nnzs
        if in_range:
          assert nnz_d[local_row] == nnz_on_diagonal
        else:
          assert nnz_od[local_row] == nnz_off_diagonal
      else:
        # No DoFs entered into this row, so ensure sparsity pattern
        # is empty.
        if in_range:
          assert nnz_d[local_row] == 0
        else:
          assert nnz_od[local_row] == 0