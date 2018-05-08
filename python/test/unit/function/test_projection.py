# Copyright (C) 2018 Nate Sime
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for projection"""

import numpy
import pytest

import dolfin
import ufl


mesh_factories = [
    (dolfin.UnitIntervalMesh, (dolfin.MPI.comm_world, 8,)),
    (dolfin.UnitSquareMesh, (dolfin.MPI.comm_world, 4, 4)),
    (dolfin.UnitCubeMesh, (dolfin.MPI.comm_world, 2, 2, 2)),
    (dolfin.UnitSquareMesh, (dolfin.MPI.comm_world, 4, 4, dolfin.CellType.Type.quadrilateral)),
    (dolfin.UnitCubeMesh, (dolfin.MPI.comm_world, 2, 2, 2, dolfin.CellType.Type.hexahedron))
]


def petsc_krylov_solver_factory(mpi_comm):
    solver = dolfin.cpp.la.PETScKrylovSolver(mpi_comm)
    solver.set_options_prefix("projection_")
    dolfin.cpp.la.PETScOptions.set("projection_ksp_type", "preonly")
    dolfin.cpp.la.PETScOptions.set("projection_pc_type", "lu")
    solver.set_from_options()
    return solver


solver_factories = [
    petsc_krylov_solver_factory,
    lambda mpi_comm: dolfin.cpp.la.PETScLUSolver(mpi_comm),
    lambda mpi_comm: None
]


def xfail_if_tensor_product_element_and_expression(mesh):
    if mesh.ufl_cell() in (ufl.quadrilateral, ufl.hexahedron):
        pytest.xfail("FIAT does not currently support evaluation of "
                     "reference bases on tensor product elements")


@pytest.mark.parametrize('solver_factory', solver_factories)
@pytest.mark.parametrize('mesh_factory', mesh_factories)
def test_project_constant(solver_factory, mesh_factory):
    func, args = mesh_factory
    mesh = func(*args)
    mesh.geometry.coord_mapping = dolfin.fem.create_coordinate_map(mesh)

    solver = solver_factory(mesh.mpi_comm())

    V = dolfin.FunctionSpace(mesh, "CG", 1)

    f1 = dolfin.project(dolfin.Constant(1.0), V, solver=solver)
    assert numpy.all(numpy.abs(f1.vector().get_local() - 1.0) < 1e-12)


@pytest.mark.parametrize('solver_factory', solver_factories)
@pytest.mark.parametrize('mesh_factory', mesh_factories)
def test_project_spatial_coordinate(solver_factory, mesh_factory):
    func, args = mesh_factory
    mesh = func(*args)
    mesh.geometry.coord_mapping = dolfin.fem.create_coordinate_map(mesh)

    solver = solver_factory(mesh.mpi_comm())

    V = dolfin.FunctionSpace(mesh, "CG", 1)

    x = ufl.SpatialCoordinate(mesh)
    f = dolfin.project(x[0], V, solver=solver)
    f_arr = f.vector().get_local()

    d2v = dolfin.dof_to_vertex_map(V)
    f_comp = numpy.array([v.point().array()[0]
                          for v in dolfin.Vertices(mesh, dolfin.cpp.mesh.MeshRangeType.ALL)],
                         dtype=numpy.double)
    f_comp = f_comp[d2v]
    ownership_range = V.dofmap().ownership_range()
    owned_offset = ownership_range[1] - ownership_range[0]
    f_comp = f_comp[:owned_offset]

    assert numpy.allclose(f_comp - f_arr, 0.0)


@pytest.mark.parametrize('solver_factory', solver_factories)
@pytest.mark.parametrize('mesh_factory', mesh_factories)
def test_project_expression(solver_factory, mesh_factory):
    func, args = mesh_factory
    mesh = func(*args)
    mesh.geometry.coord_mapping = dolfin.fem.create_coordinate_map(mesh)

    xfail_if_tensor_product_element_and_expression(mesh)

    solver = solver_factory(mesh.mpi_comm())

    V = dolfin.FunctionSpace(mesh, "CG", 1)

    x = dolfin.Expression("x[0]", degree=1)
    f = dolfin.project(x, V, solver=solver)
    f_arr = f.vector().get_local()

    d2v = dolfin.dof_to_vertex_map(V)
    f_comp = numpy.array([v.point().array()[0]
                          for v in dolfin.Vertices(mesh, dolfin.cpp.mesh.MeshRangeType.ALL)],
                         dtype=numpy.double)
    f_comp = f_comp[d2v]
    ownership_range = V.dofmap().ownership_range()
    owned_offset = ownership_range[1] - ownership_range[0]
    f_comp = f_comp[:owned_offset]

    assert numpy.allclose(f_comp - f_arr, 0.0)


@pytest.mark.parametrize('solver_factory', solver_factories)
@pytest.mark.parametrize('mesh_factory', mesh_factories)
def test_project_function(solver_factory, mesh_factory):
    func, args = mesh_factory
    mesh = func(*args)
    mesh.geometry.coord_mapping = dolfin.fem.create_coordinate_map(mesh)

    xfail_if_tensor_product_element_and_expression(mesh)

    solver = solver_factory(mesh.mpi_comm())

    V = dolfin.FunctionSpace(mesh, "CG", 1)

    f_comp = dolfin.Function(V)
    f_comp.vector()[:] = numpy.arange(f_comp.vector().get_local().shape[0],
                                      dtype=numpy.double)
    f = dolfin.project(f_comp, V, solver=solver)
    f_arr = f.vector().get_local()

    assert numpy.allclose(f_comp.vector().get_local() - f_arr, 0.0)