# Copyright (C) 2011 Johan Hake
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import (MPI, Cells, FacetRange, MeshFunction, MeshValueCollection,
                    UnitSquareMesh, VertexRange, cpp)


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
    mesh.create_connectivity(2, 1)
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
    mesh.create_connectivity(2, 0)
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
        f.values[cell.index()] = ncells - cell.index()

    g = MeshValueCollection("int", mesh, 2)
    g.assign(f)
    assert ncells == len(f.values)
    assert ncells == g.size()

    f2 = MeshFunction("int", mesh, g, 0)

    for cell in Cells(mesh):
        value = ncells - cell.index()
        assert value == g.get_value(cell.index(), 0)
        assert f2.values[cell.index()] == g.get_value(cell.index(), 0)

    h = MeshValueCollection("int", mesh, 2)
    global_indices = mesh.topology.global_indices(2)
    ncells_global = mesh.num_entities_global(2)
    for cell in Cells(mesh):
        if global_indices[cell.index()] in [5, 8, 10]:
            continue
        value = ncells_global - global_indices[cell.index()]
        h.set_value(cell.index(), int(value))

    f3 = MeshFunction("int", mesh, h, 0)

    values = f3.values
    values[values > ncells_global] = 0.

    assert MPI.sum(mesh.mpi_comm(), values.sum() * 1.0) == 140.


def test_mesh_function_assign_2D_facets():
    mesh = UnitSquareMesh(MPI.comm_world, 3, 3)
    mesh.create_entities(1)
    tdim = mesh.topology.dim
    f = MeshFunction("int", mesh, tdim - 1, 25)
    for cell in Cells(mesh):
        for i, facet in enumerate(FacetRange(cell)):
            assert 25 == f.values[facet.index()]

    g = MeshValueCollection("int", mesh, 1)
    g.assign(f)
    assert mesh.num_entities(tdim - 1) == len(f.values)
    assert mesh.num_cells() * 3 == g.size()
    for cell in Cells(mesh):
        for i, facet in enumerate(FacetRange(cell)):
            assert 25 == g.get_value(cell.index(), i)

    f2 = MeshFunction("int", mesh, g, 0)

    for cell in Cells(mesh):
        for i, facet in enumerate(FacetRange(cell)):
            assert f2.values[facet.index()] == g.get_value(cell.index(), i)


def test_mesh_function_assign_2D_vertices():
    mesh = UnitSquareMesh(MPI.comm_world, 3, 3)
    mesh.create_entities(0)
    f = MeshFunction("int", mesh, 0, 25)
    g = MeshValueCollection("int", mesh, 0)
    g.assign(f)
    assert mesh.num_entities(0) == len(f.values)
    assert mesh.num_cells() * 3 == g.size()

    f2 = MeshFunction("int", mesh, g, 0)

    for cell in Cells(mesh):
        for i, vert in enumerate(VertexRange(cell)):
            assert 25 == g.get_value(cell.index(), i)
            assert f2.values[vert.index()] == g.get_value(cell.index(), i)


def test_mvc_construction_array():
    import pygmsh

    geom = pygmsh.opencascade.Geometry()

    mesh_ele_size = 0.5
    p0 = geom.add_point([0, 0, 0], lcar=mesh_ele_size)
    p1 = geom.add_point([1, 0, 0], lcar=mesh_ele_size)
    p2 = geom.add_point([1, 1, 0], lcar=mesh_ele_size)
    p3 = geom.add_point([0, 1, 0], lcar=mesh_ele_size)

    l0 = geom.add_line(p0, p1)
    l1 = geom.add_line(p1, p2)
    l2 = geom.add_line(p2, p3)
    l3 = geom.add_line(p3, p0)

    ll = geom.add_line_loop(lines=[l0, l1, l2, l3])
    ps = geom.add_plane_surface(ll)

    # Tag line and surface
    geom.add_physical(l3, label="LINE")
    geom.add_physical(ps, label="SURFACE")

    pygmsh_mesh = pygmsh.generate_mesh(geom)
    points, cells, cell_data = pygmsh_mesh.points, pygmsh_mesh.cells, pygmsh_mesh.cell_data

    mesh = cpp.mesh.Mesh(MPI.comm_world,
                         cpp.mesh.CellType.Type.triangle, points,
                         cells['triangle'], [], cpp.mesh.GhostMode.none)
    assert mesh.degree() == 1
    assert mesh.geometry.dim == 3
    assert mesh.topology.dim == 2

    f = MeshValueCollection("size_t", mesh, 1, cells["line"], cell_data["line"]['gmsh:physical'])
    assert f.get_value(1, 1) == 1
