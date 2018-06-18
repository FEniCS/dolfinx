# Copyright (C) 2011 Johan Hake
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import UnitSquareMesh, MeshValueCollection, Cells, MPI, MeshFunction, FacetRange, VertexRange
from dolfin.log import info


def test_assign_2D_cells():
    mesh = UnitSquareMesh(MPI.comm_world, 3, 3)
    ncells = mesh.num_cells()
    f = MeshValueCollection("int", mesh, 2)
    all_new = True
    for cell in Cells(mesh):
        value = ncells - cell.index()
        all_new = all_new and f.set_value(cell.index(), value)
    g = MeshValueCollection("int", mesh, 2)
    g.assign(f)
    assert ncells == f.size()
    assert ncells == g.size()
    assert all_new

    for cell in Cells(mesh):
        value = ncells - cell.index()
        assert value, g.get_value(cell.index() == 0)

    old_value = g.get_value(0, 0)
    g.set_value(0, 0, old_value + 1)
    assert old_value + 1 == g.get_value(0, 0)


def test_assign_2D_facets():
    mesh = UnitSquareMesh(MPI.comm_world, 3, 3)
    mesh.init(2, 1)
    ncells = mesh.num_cells()
    f = MeshValueCollection("int", mesh, 1)
    all_new = True
    for cell in Cells(mesh):
        value = ncells - cell.index()
        for i, facet in enumerate(FacetRange(cell)):
            all_new = all_new and f.set_value(cell.index(), i, value + i)

    g = MeshValueCollection("int", mesh, 1)
    g.assign(f)
    assert ncells * 3 == f.size()
    assert ncells * 3 == g.size()
    assert all_new

    for cell in Cells(mesh):
        value = ncells - cell.index()
        for i, facet in enumerate(FacetRange(cell)):
            assert value + i == g.get_value(cell.index(), i)


def test_assign_2D_vertices():
    mesh = UnitSquareMesh(MPI.comm_world, 3, 3)
    mesh.init(2, 0)
    ncells = mesh.num_cells()
    f = MeshValueCollection("int", mesh, 0)
    all_new = True
    for cell in Cells(mesh):
        value = ncells - cell.index()
        for i, vert in enumerate(VertexRange(cell)):
            all_new = all_new and f.set_value(cell.index(), i, value + i)

    g = MeshValueCollection("int", mesh, 0)
    g.assign(f)
    assert ncells * 3 == f.size()
    assert ncells * 3 == g.size()
    assert all_new

    for cell in Cells(mesh):
        value = ncells - cell.index()
        for i, vert in enumerate(VertexRange(cell)):
            assert value + i == g.get_value(cell.index(), i)


def test_mesh_function_assign_2D_cells():
    mesh = UnitSquareMesh(MPI.comm_world, 3, 3)
    ncells = mesh.num_cells()
    f = MeshFunction("int", mesh, mesh.topology.dim, 0)
    for cell in Cells(mesh):
        f[cell] = ncells - cell.index()

    g = MeshValueCollection("int", mesh, 2)
    g.assign(f)
    assert ncells == f.size()
    assert ncells == g.size()

    f2 = MeshFunction("int", mesh, g, 0)

    for cell in Cells(mesh):
        value = ncells - cell.index()
        assert value == g.get_value(cell.index(), 0)
        assert f2[cell] == g.get_value(cell.index(), 0)

    h = MeshValueCollection("int", mesh, 2)
    global_indices = mesh.topology.global_indices(2)
    ncells_global = mesh.num_entities_global(2)
    for cell in Cells(mesh):
        if global_indices[cell.index()] in [5, 8, 10]:
            continue
        value = ncells_global - global_indices[cell.index()]
        h.set_value(cell.index(), int(value))

    f3 = MeshFunction("int", mesh, h, 0)

    values = f3.array()
    values[values > ncells_global] = 0.

    info(str(values))
    info(str(values.sum()))

    assert MPI.sum(mesh.mpi_comm(), values.sum() * 1.0) == 140.


def test_mesh_function_assign_2D_facets():
    mesh = UnitSquareMesh(MPI.comm_world, 3, 3)
    mesh.init(1)
    f = MeshFunction("int", mesh, mesh.topology.dim - 1, 25)
    for cell in Cells(mesh):
        for i, facet in enumerate(FacetRange(cell)):
            assert 25 == f[facet]

    g = MeshValueCollection("int", mesh, 1)
    g.assign(f)
    assert mesh.num_facets() == f.size()
    assert mesh.num_cells() * 3 == g.size()
    for cell in Cells(mesh):
        for i, facet in enumerate(FacetRange(cell)):
            assert 25 == g.get_value(cell.index(), i)

    f2 = MeshFunction("int", mesh, g, 0)

    for cell in Cells(mesh):
        for i, facet in enumerate(FacetRange(cell)):
            assert f2[facet] == g.get_value(cell.index(), i)


def test_mesh_function_assign_2D_vertices():
    mesh = UnitSquareMesh(MPI.comm_world, 3, 3)
    mesh.init(0)
    f = MeshFunction("int", mesh, 0, 25)
    g = MeshValueCollection("int", mesh, 0)
    g.assign(f)
    assert mesh.num_vertices() == f.size()
    assert mesh.num_cells() * 3 == g.size()

    f2 = MeshFunction("int", mesh, g, 0)

    for cell in Cells(mesh):
        for i, vert in enumerate(VertexRange(cell)):
            assert 25 == g.get_value(cell.index(), i)
            assert f2[vert] == g.get_value(cell.index(), i)
