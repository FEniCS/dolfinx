# Copyright (C) 2020 Joseph P. Dean
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfinx
import ufl
import numpy as np


def test_locate_dofs_geometrical():
    """Test that locate_dofs_geometrical when passed two function
    spaces returns the correct degrees of freedom in each space.
    """
    mesh = dolfinx.generation.UnitSquareMesh(dolfinx.MPI.comm_world, 4, 8)
    p0, p1 = 1, 2
    P0 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p0)
    P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), p1)

    W = dolfinx.function.FunctionSpace(mesh, P0 * P1)
    V = W.sub(0).collapse()

    dofs = dolfinx.fem.locate_dofs_geometrical(
        (W.sub(0), V), lambda x: np.isclose(x.T, [0, 0, 0]).all(axis=1))

    # Collect dofs from all processes (does not matter that the numbering
    # is local to each process for this test)
    all_dofs = np.vstack(dolfinx.MPI.comm_world.allgather(dofs))

    # Check only one dof pair is returned
    assert len(all_dofs) == 1

    # On process with the dof pair
    if len(dofs) == 1:
        # Check correct dof returned in W
        coords_W = W.tabulate_dof_coordinates()
        assert np.isclose(coords_W[dofs[0][0]], [0, 0, 0]).all()
        # Check correct dof returned in V
        coords_V = V.tabulate_dof_coordinates()
        assert np.isclose(coords_V[dofs[0][1]], [0, 0, 0]).all()
