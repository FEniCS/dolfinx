# Copyright (C) 2018 Michal Habera
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfinx import cpp as _cpp


class DofMap:
    """Degree-of-freedom map

    This class handles the mapping of degrees of freedom. It builds
    a dof map based on a FiniteElement on a specific mesh.
    """

    _cpp_object: _cpp.fem.DofMap

    def __init__(self, dofmap: _cpp.fem.DofMap):
        self._cpp_object = dofmap

    def cell_dofs(self, cell_index: int):
        """Cell local-global dof map

        Args:
            cell: The cell index.

        Returns:
            Local-global dof map for the cell (using process-local indices).
        """
        return self._cpp_object.cell_dofs(cell_index)

    @property
    def bs(self):
        """Returns the block size of the dofmap"""
        return self._cpp_object.bs

    @property
    def dof_layout(self):
        """Layout of dofs on an element."""
        return self._cpp_object.dof_layout

    @property
    def index_map(self):
        """Index map that described the parallel distribution of the dofmap."""
        return self._cpp_object.index_map

    @property
    def index_map_bs(self):
        """Block size of the index map."""
        return self._cpp_object.index_map_bs

    @property
    def list(self):
        """Adjacency list with dof indices for each cell."""
        return self._cpp_object.map()
