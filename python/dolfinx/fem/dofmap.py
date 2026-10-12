# Copyright (C) 2018-2025 Michal Habera and Paul T. Kühner
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Degree-of-freedom maps."""

import functools
import typing
from collections.abc import Sequence

from mpi4py.MPI import Comm

import numpy as np
import numpy.typing as npt

from dolfinx import cpp as _cpp
from dolfinx.common import IndexMap
from dolfinx.cpp.fem import DofMap as _DofMap
from dolfinx.cpp.fem import create_dofmaps as _create_dofmaps
from dolfinx.fem.element import ElementDofLayout, FiniteElement
from dolfinx.graph import AdjacencyList

if typing.TYPE_CHECKING:
    import dolfinx.mesh


class DofMap:
    """Degree-of-freedom map.

    This class handles the mapping of degrees of freedom. It builds a
    dof map based on a FiniteElement on a specific mesh.
    """

    _cpp_object: _DofMap

    def __init__(self, dofmap: _DofMap):
        """Initialise a degree-of-freedom map."""
        self._cpp_object = dofmap

    def __eq__(self, other: object) -> bool:
        """Check that two wrappers hold the same dofmap."""
        if not isinstance(other, DofMap):
            return NotImplemented
        return self._cpp_object == other._cpp_object

    def __hash__(self) -> int:
        """Hash of the wrapped dofmap."""
        return hash(self._cpp_object)

    def cell_dofs(self, cell_index: int) -> npt.NDArray[np.int32]:
        """Cell local-global dof map.

        Args:
            cell_index: The cell index.

        Returns:
            Local-global dof map for the cell (using process-local
            indices).
        """
        return self._cpp_object.cell_dofs(cell_index)

    @property
    def bs(self) -> int:
        """Block size of the dofmap."""
        return self._cpp_object.bs

    @functools.cached_property
    def dof_layout(self) -> ElementDofLayout:
        """Layout of dofs on an element.

        Note:
            This is a cached property. The wrapper is built on first
            access and the same object is returned thereafter.
        """
        return ElementDofLayout(self._cpp_object.dof_layout)

    @functools.cached_property
    def index_map(self) -> IndexMap:
        """Index map describing parallel distribution of the dofmap.

        Note:
            This is a cached property. The wrapper is built on first
            access and the same object is returned thereafter.
        """
        return IndexMap(self._cpp_object.index_map)

    @property
    def index_map_bs(self) -> int:
        """Block size of the index map."""
        return self._cpp_object.index_map_bs

    @property
    def list(self) -> npt.NDArray[np.int32]:
        """Adjacency list with dof indices for each cell."""
        return self._cpp_object.map()


def create_dofmaps(
    comm: Comm, topology: "dolfinx.mesh.Topology", elements: Sequence[FiniteElement]
) -> list[DofMap]:
    """Create degree-of-freedom maps on a given topology.

    Args:
        comm: MPI communicator
        topology: Mesh topology
        elements: Sequence of elements

    Returns:
        List of degree-of-freedom maps where the ``i``-th map is the map
        for ``elements[i]``.
    """
    elements_cpp = [e._cpp_object for e in elements]
    cpp_dofmaps = _create_dofmaps(comm, topology._cpp_object, elements_cpp)  # type: ignore[arg-type]
    return [DofMap(cpp_object) for cpp_object in cpp_dofmaps]


def transpose_dofmap(dofmap: npt.NDArray[np.int32], num_cells: int) -> AdjacencyList[np.int32]:
    """Build the index to ``(cell, local index)`` map from a dofmap.

    Args:
        dofmap: Dofmap ``(cell, local index) -> index``, with shape
            ``(num_cells, dofs_per_cell)``.
        num_cells: Number of cells in ``dofmap`` to consider. Cells
            beyond ``num_cells`` are ignored.

    Returns:
        Adjacency list where node ``i`` holds the positions in the
        flattened ``dofmap`` at which index ``i`` appears.
    """
    return AdjacencyList(_cpp.fem.transpose_dofmap(dofmap, num_cells))
