# Copyright (C) 2011 Johan Hake
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import MPI, MeshFunction, MeshValueCollection, UnitSquareMesh, cpp


def test_assign_2D_cells():
    mesh = UnitSquareMesh(MPI.comm_world, 3, 3)
    ncells = mesh.num_cells()
    f = MeshValueCollection("int", mesh, 2)
    all_new = True
    for c in range(ncells):
        value = ncells - c
        all_new = all_new and f.set_value(c, value)
    g = MeshValueCollection("int", mesh, 2)
    g.assign(f)
    assert ncells == f.size()
    assert ncells == g.size()
    assert all_new

    for c in range(ncells):
        value = ncells - c
        assert value, g.get_value(c == 0)

    old_value = g.get_value(0, 0)
    g.set_value(0, 0, old_value + 1)
    assert old_value + 1 == g.get_value(0, 0)


def test_assign_2D_facets():
    mesh = UnitSquareMesh(MPI.comm_world, 3, 3)
    mesh.create_connectivity(2, 1)
    tdim = mesh.topology.dim
    num_cell_facets = cpp.mesh.cell_num_entities(mesh.cell_type, tdim - 1)
    ncells = mesh.num_cells()

    f = MeshValueCollection("int", mesh, 1)
    all_new = True
    for c in range(ncells):
        value = ncells - c
        for i in range(num_cell_facets):
            all_new = all_new and f.set_value(c, i, value + i)

    g = MeshValueCollection("int", mesh, 1)
    g.assign(f)
    assert ncells * 3 == f.size()
    assert ncells * 3 == g.size()
    assert all_new

    for c in range(ncells):
        value = ncells - c
        for i in range(num_cell_facets):
            assert value + i == g.get_value(c, i)


def test_assign_2D_vertices():
    mesh = UnitSquareMesh(MPI.comm_world, 3, 3)
    mesh.create_connectivity(2, 0)
    ncells = mesh.num_cells()
    num_cell_vertices = cpp.mesh.cell_num_vertices(mesh.cell_type)

    f = MeshValueCollection("int", mesh, 0)
    all_new = True
    for c in range(ncells):
        value = ncells - c
        for i in range(num_cell_vertices):
            all_new = all_new and f.set_value(c, i, value + i)

    g = MeshValueCollection("int", mesh, 0)
    g.assign(f)
    assert ncells * 3 == f.size()
    assert ncells * 3 == g.size()
    assert all_new

    for c in range(ncells):
        value = ncells - c
        for i in range(num_cell_vertices):
            assert value + i == g.get_value(c, i)


def test_mesh_function_assign_2D_cells():
    mesh = UnitSquareMesh(MPI.comm_world, 3, 3)
    ncells = mesh.num_cells()
    f = MeshFunction("int", mesh, mesh.topology.dim, 0)
    for c in range(ncells):
        f.values[c] = ncells - c

    g = MeshValueCollection("int", mesh, 2)
    g.assign(f)
    assert ncells == len(f.values)
    assert ncells == g.size()

    f2 = MeshFunction("int", mesh, g, 0)

    for c in range(mesh.num_cells()):
        value = ncells - c
        assert value == g.get_value(c, 0)
        assert f2.values[c] == g.get_value(c, 0)

    h = MeshValueCollection("int", mesh, 2)
    global_indices = mesh.topology.global_indices(2)
    ncells_global = mesh.num_entities_global(2)
    for c in range(mesh.num_cells()):
        if global_indices[c] in [5, 8, 10]:
            continue
        value = ncells_global - global_indices[c]
        h.set_value(c, int(value))

    f3 = MeshFunction("int", mesh, h, 0)

    values = f3.values
    values[values > ncells_global] = 0.

    assert MPI.sum(mesh.mpi_comm(), values.sum() * 1.0) == 140.


def test_mesh_function_assign_2D_facets():
    mesh = UnitSquareMesh(MPI.comm_world, 3, 3)
    mesh.create_entities(1)
    tdim = mesh.topology.dim
    num_cell_facets = cpp.mesh.cell_num_entities(mesh.cell_type, tdim - 1)

    f = MeshFunction("int", mesh, tdim - 1, 25)
    connectivity = mesh.topology.connectivity(tdim, tdim - 1)
    for c in range(mesh.num_cells()):
        facets = connectivity.connections(c)
        for i in range(num_cell_facets):
            assert 25 == f.values[facets[i]]

    g = MeshValueCollection("int", mesh, 1)
    g.assign(f)
    assert mesh.num_entities(tdim - 1) == len(f.values)
    assert mesh.num_cells() * 3 == g.size()
    for c in range(mesh.num_cells()):
        for i in range(num_cell_facets):
            assert 25 == g.get_value(c, i)

    f2 = MeshFunction("int", mesh, g, 0)

    connectivity = mesh.topology.connectivity(tdim, tdim - 1)
    for c in range(mesh.num_cells()):
        facets = connectivity.connections(c)
        for i in range(num_cell_facets):
            assert f2.values[facets[i]] == g.get_value(c, i)


def test_mesh_function_assign_2D_vertices():
    mesh = UnitSquareMesh(MPI.comm_world, 3, 3)
    mesh.create_entities(0)
    f = MeshFunction("int", mesh, 0, 25)
    g = MeshValueCollection("int", mesh, 0)
    g.assign(f)
    assert mesh.num_entities(0) == len(f.values)
    assert mesh.num_cells() * 3 == g.size()

    f2 = MeshFunction("int", mesh, g, 0)

    num_cell_vertices = cpp.mesh.cell_num_vertices(mesh.cell_type)
    tdim = mesh.topology.dim
    connectivity = mesh.topology.connectivity(tdim, 0)
    for c in range(mesh.num_cells()):
        vertices = connectivity.connections(c)
        for i in range(num_cell_vertices):
            assert 25 == g.get_value(c, i)
            assert f2.values[vertices[i]] == g.get_value(c, i)


def test_mvc_construction_array_tet_tri():
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

    geom.set_transfinite_surface(ps, size=[4, 4])
    box = geom.extrude(ps, [0, 0, 1], num_layers=4)

    # Tag line and surface
    geom.add_physical([p0, p3], label="POINT_LEFT")
    geom.add_physical([p1, p2], label="POINT_RIGHT")
    geom.add_physical([l0, l2], label="LINE_X")
    geom.add_physical([l1, l3], label="LINE_Y")
    geom.add_physical(ps, label="SURFACE")
    geom.add_physical(box[1], label="BOX")

    pygmsh_mesh = pygmsh.generate_mesh(geom)
    points, cells, cell_data = (
        pygmsh_mesh.points,
        pygmsh_mesh.cells,
        pygmsh_mesh.cell_data,
    )

    mesh = cpp.mesh.Mesh(
        MPI.comm_world,
        cpp.mesh.CellType.tetrahedron,
        points,
        cells["tetra"],
        [],
        cpp.mesh.GhostMode.none,
    )
    assert mesh.degree() == 1
    assert mesh.geometry.dim == 3
    assert mesh.topology.dim == 3

    mvc_vertex = MeshValueCollection(
        "size_t", mesh, 0, cells["vertex"],
        cell_data["vertex"]["gmsh:physical"]
    )
    assert mvc_vertex.get_value(0, 0) == 1

    mvc_line = MeshValueCollection(
        "size_t", mesh, 1, cells["line"], cell_data["line"]["gmsh:physical"]
    )
    assert mvc_line.get_value(0, 4) == 4

    mvc_triangle = MeshValueCollection(
        "size_t", mesh, 2, cells["triangle"],
        cell_data["triangle"]["gmsh:physical"]
    )
    assert mvc_triangle.get_value(0, 3) == 5

    mvc_tetra = MeshValueCollection(
        "size_t", mesh, 3, cells["tetra"], cell_data["tetra"]["gmsh:physical"]
    )
    assert mvc_tetra.get_value(0, 0) == 6


def test_mvc_construction_array_hex_quad():
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

    geom.set_transfinite_surface(ps, size=[4, 4])
    geom.add_raw_code("Recombine Surface {%s};" % ps.id)
    box = geom.extrude(ps, [0, 0, 1], num_layers=4, recombine=True)

    # Tag line, surface and volume
    geom.add_physical([p0, p3], label="POINT_LEFT")
    geom.add_physical([p1, p2], label="POINT_RIGHT")
    geom.add_physical([l0, l2], label="LINE_X")
    geom.add_physical([l1, l3], label="LINE_Y")
    geom.add_physical(ps, label="SURFACE")
    geom.add_physical(box[1], label="BOX")

    pygmsh_mesh = pygmsh.generate_mesh(geom)
    points, cells, cell_data = (
        pygmsh_mesh.points,
        pygmsh_mesh.cells,
        pygmsh_mesh.cell_data,
    )

    mesh = cpp.mesh.Mesh(
        MPI.comm_world,
        cpp.mesh.CellType.hexahedron,
        points,
        cells["hexahedron"],
        [],
        cpp.mesh.GhostMode.none,
    )

    assert mesh.degree() == 1
    assert mesh.geometry.dim == 3
    assert mesh.topology.dim == 3

    mvc_vertex = MeshValueCollection(
        "size_t", mesh, 0, cells["vertex"],
        cell_data["vertex"]["gmsh:physical"]
    )
    assert mvc_vertex.get_value(0, 0) == 1

    mvc_line = MeshValueCollection(
        "size_t", mesh, 1, cells["line"], cell_data["line"]["gmsh:physical"]
    )
    mvc_line.values()
    assert mvc_line.get_value(0, 0) == 3

    mvc_quad = MeshValueCollection(
        "size_t", mesh, 2, cells["quad"], cell_data["quad"]["gmsh:physical"]
    )
    assert mvc_quad.get_value(0, 0) == 5

    mvc_hexa = MeshValueCollection(
        "size_t", mesh, 3, cells["hexahedron"],
        cell_data["hexahedron"]["gmsh:physical"]
    )
    assert mvc_hexa.get_value(0, 0) == 6
