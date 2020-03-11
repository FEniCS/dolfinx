# Copyright (C) 2013 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from math import pi

import pytest

from dolfinx import MPI, MeshQuality, UnitCubeMesh


@pytest.mark.skip(reason="Needs updating for geometry")
def test_dihedral_angles_min_max():
    # Create 3D mesh with regular tetrahedron
    mesh = UnitCubeMesh(MPI.comm_world, 2, 2, 2)
    dang_min, dang_max = MeshQuality.dihedral_angles_min_max(mesh)
    assert round(dang_min * (180 / pi) - 45.0) == 0
    assert round(dang_max * (180 / pi) - 90.0) == 0
