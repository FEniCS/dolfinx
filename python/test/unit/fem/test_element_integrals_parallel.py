# Copyright (C) 2009-2020 Garth N. Wells, Matthew W. Scroggs and Jorgen S. Dokken
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the fem interface"""


import numpy as np
import pytest
import os
from itertools import combinations, product
from dolfin import MPI, fem, FunctionSpace, Function
from ufl import dS
from dolfin.cpp.mesh import GhostMode
from dolfin.io import XDMFFile


@pytest.mark.parametrize('space_type', ["CG", "DG"])
@pytest.mark.parametrize("filename", ["UnitSquareMesh_triangle.xdmf",
                                      "UnitCubeMesh_tetra.xdmf",
                                      "UnitSquareMesh_quad.xdmf",
                                      "UnitCubeMesh_hexahedron.xdmf"])
def test_plus_minus(filename, space_type, datadir):
    """Test that ('+') and ('-') give the same value for continuous functions"""
    with XDMFFile(MPI.comm_world, os.path.join(datadir, filename)) as xdmf:
        if MPI.size(MPI.comm_world) == 1:  # Serial
            mesh = xdmf.read_mesh(GhostMode.none)
        else:
            mesh = xdmf.read_mesh(GhostMode.shared_facet)

    V = FunctionSpace(mesh, (space_type, 1))
    v = Function(V)
    v.interpolate(lambda x: x[0] - 2 * x[1])

    results = []
    # Check that these two integrals are equal
    for pm1, pm2 in product(["+", "-"], repeat=2):
        a = v(pm1) * v(pm2) * dS
        results.append(fem.assemble_scalar(a))
    print(results)
    for i, j in combinations(results, 2):
        assert np.isclose(i, j)
