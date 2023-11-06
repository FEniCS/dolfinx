# Copyright (C) 2023 Jorgen Dokken and Matthew Scroggs
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfinx
import ufl
import numpy as np
from mpi4py import MPI
import basix.ufl
import pytest


@pytest.mark.parametrize("degree", range(1, 4))
def test_default(degree):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

    CG2_vect = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    Qe = basix.ufl.quadrature_element(msh.topology.cell_name(), value_shape=(), scheme="default", degree=degree)
    Quad = dolfinx.fem.functionspace(msh, Qe)

    u = dolfinx.fem.Function(Quad)
    v = ufl.TrialFunction(CG2_vect)

    dx_m = ufl.Measure("dx", domain=msh, metadata={"quadrature_degree": 1, "quadrature_scheme": "default"})
    ds = ufl.Measure("ds", domain=msh)

    residual = u * v * dx_m
    vol = dolfinx.fem.form(residual)
    residual = v * ds
    surf = dolfinx.fem.form(residual)

    residual = u * v * dx_m + v * ds
    vol_surf = dolfinx.fem.form(residual)

    vol_v = dolfinx.fem.assemble_vector(vol)
    sur_v = dolfinx.fem.assemble_vector(surf)

    vol_surf = dolfinx.fem.assemble_vector(vol_surf)

    assert np.allclose(vol_v.array + sur_v.array, vol_surf.array)


def test_points_and_weights():
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

    CG2_vect = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    Qe = basix.ufl.quadrature_element(
        msh.topology.cell_name(), value_shape=(),
        points=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1/3, 1/3]]),
        weights=np.array([0.2, 0.2, 0.2, 0.4]),)
    Quad = dolfinx.fem.functionspace(msh, Qe)

    u = dolfinx.fem.Function(Quad)
    v = ufl.TrialFunction(CG2_vect)

    dx_m = ufl.Measure("dx", domain=msh, metadata={"quadrature_degree": 1, "quadrature_scheme": "default"})
    ds = ufl.Measure("ds", domain=msh)

    residual = u * v * dx_m
    vol = dolfinx.fem.form(residual)
    residual = v * ds
    surf = dolfinx.fem.form(residual)

    residual = u * v * dx_m + v * ds
    vol_surf = dolfinx.fem.form(residual)

    vol_v = dolfinx.fem.assemble_vector(vol)
    sur_v = dolfinx.fem.assemble_vector(surf)

    vol_surf = dolfinx.fem.assemble_vector(vol_surf)

    assert np.allclose(vol_v.array + sur_v.array, vol_surf.array)
