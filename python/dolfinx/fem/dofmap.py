# Copyright (C) 2018-2025 Michal Habera and Paul T. Kühner
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import typing
from collections.abc import Sequence

from mpi4py.MPI import Comm

from basix.finite_element import FiniteElement
from dolfinx.cpp.fem import DofMap as _DofMap
from dolfinx.cpp.fem import create_dofmaps as _create_dofmaps

if typing.TYPE_CHECKING:
    from dolfinx.mesh import Topology


class DofMap:
    """Degree-of-freedom map.

    This class handles the mapping of degrees of freedom. It builds a
    dof map based on a FiniteElement on a specific mesh.
    """

    _cpp_object: _DofMap

    def __init__(self, dofmap: _DofMap):
        self._cpp_object = dofmap

    def cell_dofs(self, cell_index: int):
        """Cell local-global dof map

        Args:
            cell: The cell index.

        Returns:
            Local-global dof map for the cell (using process-local
            indices).
        """
        return self._cpp_object.cell_dofs(cell_index)

    @property
    def bs(self):
        """Block size of the dofmap."""
        return self._cpp_object.bs

    @property
    def dof_layout(self):
        """Layout of dofs on an element."""
        return self._cpp_object.dof_layout

    @property
    def index_map(self):
        """Index map that described the parallel distribution of the
        dofmap."""
        return self._cpp_object.index_map

    @property
    def index_map_bs(self):
        """Block size of the index map."""
        return self._cpp_object.index_map_bs

    @property
    def list(self):
        """Adjacency list with dof indices for each cell."""
        return self._cpp_object.map()


def create_dofmaps(
    comm: Comm, topology: "Topology", elements: Sequence[FiniteElement]
) -> list[DofMap]:
    """Create a set of dofmaps on a given topology

    Args:
        comm: MPI communicator
        topology: Mesh topology
        elements: Sequence of coordinate elements

    Returns:
        List of new DOF maps
    """
    elements_cpp = [e._e for e in elements]
    cpp_dofmaps = _create_dofmaps(comm, topology._cpp_object, elements_cpp)
    return [DofMap(cpp_object) for cpp_object in cpp_dofmaps]
