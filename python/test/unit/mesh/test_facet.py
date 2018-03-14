# Copyright (C) 2017 Tormod Landet
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import UnitSquareMesh, Facets, MPI

def test_normal():
    "Test that the normal() method is wrapped"
    mesh = UnitSquareMesh(MPI.comm_world, 4, 4)
    mesh.init(1)
    for facet in Facets(mesh):
        n = facet.normal()
        nx, ny, nz = n[0], n[1], n[2]
        assert isinstance(nx, float)
        assert isinstance(ny, float)
        assert isinstance(nz, float)
