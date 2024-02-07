# Copyright (C) 2021 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the function library"""

from mpi4py import MPI

import numpy as np
import pytest

import ufl
from dolfinx import default_scalar_type
from dolfinx.fem import Constant, assemble_scalar, form
from dolfinx.mesh import create_unit_cube, create_unit_interval, create_unit_square


def test_facet_area1D():
    mesh = create_unit_interval(MPI.COMM_WORLD, 10)

    # NOTE: Area of a vertex is defined to 1 in ufl
    c0 = ufl.FacetArea(mesh)
    c = Constant(mesh, default_scalar_type(1))

    ds = ufl.Measure("ds", domain=mesh)
    a0 = mesh.comm.allreduce(assemble_scalar(form(c * ds)), op=MPI.SUM)
    a = mesh.comm.allreduce(assemble_scalar(form(c0 * ds)), op=MPI.SUM)
    assert np.isclose(a.real, 2)
    assert np.isclose(a0.real, 2)


@pytest.mark.parametrize(
    "mesh_factory",
    [
        (create_unit_square, (MPI.COMM_WORLD, 3, 3), 1.0 / 3),
        #   (create_unit_square,
        #   (MPI.COMM_WORLD, 3, 3, CellType.quadrilateral), 1. / 3),
        (create_unit_cube, (MPI.COMM_WORLD, 3, 3, 3), 1 / 18.0),
        #   (create_unit_cube,
        #   (MPI.COMM_WORLD, 3, 3, 3, CellType.hexahedron), 1. / 9)
    ],
)
def test_facet_area(mesh_factory):
    """Compute facet area of cell. UFL currently only supports affine
    cells for this computation"""
    # NOTE: UFL only supports facet area calculations of affine cells
    func, args, exact_area = mesh_factory
    mesh = func(*args)
    c0 = ufl.FacetArea(mesh)
    c = Constant(mesh, default_scalar_type(1))
    tdim = mesh.topology.dim
    num_faces = 4 if tdim == 2 else 6

    ds = ufl.Measure("ds", domain=mesh)
    a = mesh.comm.allreduce(assemble_scalar(form(c * ds)), op=MPI.SUM)
    a0 = mesh.comm.allreduce(assemble_scalar(form(c0 * ds)), op=MPI.SUM)
    assert np.isclose(a.real, num_faces)
    assert np.isclose(a0.real, num_faces * exact_area)
