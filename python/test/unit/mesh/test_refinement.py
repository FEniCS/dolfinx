# Copyright (C) 2018 Chris N Richardson
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfinx import UnitSquareMesh, UnitCubeMesh, MPI, FunctionSpace
from dolfinx.cpp.refinement import refine


def test_RefineUnitSquareMesh():
    """Refine mesh of unit square."""
    mesh = UnitSquareMesh(MPI.comm_world, 5, 7)
    mesh.create_entities(1)
    mesh = refine(mesh, False)
    assert mesh.num_entities_global(0) == 165
    assert mesh.num_entities_global(2) == 280


def test_RefineUnitCubeMesh_repartition():
    """Refine mesh of unit cube."""
    mesh = UnitCubeMesh(MPI.comm_world, 5, 7, 9)
    mesh.create_entities(1)
    mesh = refine(mesh, True)
    assert mesh.num_entities_global(0) == 3135
    assert mesh.num_entities_global(3) == 15120
    Q = FunctionSpace(mesh, ("CG", 1))
    assert(Q)


def test_RefineUnitCubeMesh_keep_partition():
    """Refine mesh of unit cube."""
    mesh = UnitCubeMesh(MPI.comm_world, 5, 7, 9)
    mesh.create_entities(1)
    mesh = refine(mesh, False)
    assert mesh.num_entities_global(0) == 3135
    assert mesh.num_entities_global(3) == 15120
    Q = FunctionSpace(mesh, ("CG", 1))
    assert(Q)


def test_refinement_gdim():
    """Test that 2D refinement is still 2D"""
    mesh = UnitSquareMesh(MPI.comm_world, 3, 4)
    mesh2 = refine(mesh, True)
    assert mesh.geometry.dim == mesh2.geometry.dim
