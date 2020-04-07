# Copyright (C) 2009-2020 Garth N. Wells, Matthew W. Scroggs and Jorgen S. Dokken
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the fem interface"""

import numpy as np
import pytest
from dolfinx_utils.test.skips import skip_in_parallel

from dolfinx import MPI, fem, FunctionSpace
from ufl import dx, TestFunction
from dolfinx.cpp.mesh import CellType

from dolfinx import UnitSquareMesh
from dolfinx.fem import assemble_vector


@skip_in_parallel
@pytest.mark.parametrize('space_type', ["RTCF"])
@pytest.mark.parametrize('degree', [1, 2, 3])
def test_components_Hdiv(degree, space_type):
    """Test that the two components of the vector are correct."""
    mesh = UnitSquareMesh(MPI.comm_world, 1, 1, CellType.quadrilateral)
    mesh.geometry.coord_mapping = fem.create_coordinate_map(mesh)
    mesh.create_connectivity_all()

    V = FunctionSpace(mesh, (space_type, degree))

    v = TestFunction(V)
    dofmap = V.mesh.geometry.dofmap().links(0)
    v_of_e = V.mesh.topology.connectivity(1, 0)
    e_of_cell = V.mesh.topology.connectivity(2, 1).links(0)
    cell_dofs = V.dofmap.cell_dofs(0)
    for component in [0, 1]:
        b = assemble_vector(v[component] * dx)
        for edge_n, edge in enumerate(e_of_cell):
            points = v_of_e.links(edge)
            coords = [V.mesh.geometry.x[dofmap[i]] for i in points]
            mid = (coords[0] + coords[1]) / 2
            if np.isclose(mid[component], 1) or np.isclose(mid[component], 0):
                for dof in V.dofmap.dof_layout.entity_dofs(1, edge_n):
                    assert not np.isclose(b[cell_dofs[dof]], 0)
            else:
                for dof in V.dofmap.dof_layout.entity_dofs(1, edge_n):
                    assert np.isclose(b[cell_dofs[dof]], 0)


@skip_in_parallel
@pytest.mark.parametrize('space_type', ["RTCE"])
@pytest.mark.parametrize('degree', [1, 2, 3])
def test_components_Hcurl(degree, space_type):
    """Test that the two components of the vector are correct."""
    mesh = UnitSquareMesh(MPI.comm_world, 1, 1, CellType.quadrilateral)
    mesh.geometry.coord_mapping = fem.create_coordinate_map(mesh)
    mesh.create_connectivity_all()

    V = FunctionSpace(mesh, (space_type, degree))

    v = TestFunction(V)
    dofmap = V.mesh.geometry.dofmap().links(0)
    v_of_e = V.mesh.topology.connectivity(1, 0)
    e_of_cell = V.mesh.topology.connectivity(2, 1).links(0)
    cell_dofs = V.dofmap.cell_dofs(0)
    for component in [0, 1]:
        b = assemble_vector(v[component] * dx)
        for edge_n, edge in enumerate(e_of_cell):
            points = v_of_e.links(edge)
            coords = [V.mesh.geometry.x[dofmap[i]] for i in points]
            mid = (coords[0] + coords[1]) / 2
            if np.isclose(mid[component], 1) or np.isclose(mid[component], 0):
                for dof in V.dofmap.dof_layout.entity_dofs(1, edge_n):
                    assert np.isclose(b[cell_dofs[dof]], 0)
            else:
                for dof in V.dofmap.dof_layout.entity_dofs(1, edge_n):
                    assert not np.isclose(b[cell_dofs[dof]], 0)
