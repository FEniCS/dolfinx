# Copyright (C) 2006-2011 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest
import numpy
from dolfin import UnitCubeMesh, MPI, Vertices, Edges, Faces, Facets, Cells, VertexRange


def test_vertex_iterators():
    "Iterate over vertices"

    mesh = UnitCubeMesh(MPI.comm_world, 5, 5, 5)
    for i in range(4):
        mesh.init(0, i)

    # Test connectivity
    cons = [(i, mesh.topology.connectivity(0, i)) for i in range(4)]

    # Test writability
    for i, con in cons:
        def assign(con, i):
            con(i)[0] = 1
        with pytest.raises(Exception):
            assign(con, i)

    n = 0
    for i, v in enumerate(Vertices(mesh)):
        n += 1
        for j, con in cons:
            assert numpy.all(con(i) == v.entities(j))

    assert n == mesh.num_vertices()


def test_edge_iterators():
    "Iterate over edges"

    mesh = UnitCubeMesh(MPI.comm_world, 5, 5, 5)
    for i in range(4):
        mesh.init(1, i)

    # Test connectivity
    cons = [(i, mesh.topology.connectivity(1, i)) for i in range(4)]

    # Test writability
    for i, con in cons:
        def assign(con, i):
            con(i)[0] = 1
        with pytest.raises(Exception):
            assign(con, i)

    n = 0
    for i, e in enumerate(Edges(mesh)):
        n += 1
        for j, con in cons:
            assert numpy.all(con(i) == e.entities(j))

    assert n == mesh.num_entities(1)


def test_face_iterator():
    "Iterate over faces"

    mesh = UnitCubeMesh(MPI.comm_world, 5, 5, 5)
    for i in range(4):
        mesh.init(2, i)

    # Test connectivity
    cons = [(i, mesh.topology.connectivity(2, i)) for i in range(4)]

    # Test writability
    for i, con in cons:
        def assign(con, i):
            con(i)[0] = 1
        with pytest.raises(Exception):
            assign(con, i)

    n = 0
    for i, f in enumerate(Faces(mesh)):
        n += 1
        for j, con in cons:
            assert numpy.all(con(i) == f.entities(j))

    assert n == mesh.num_entities(2)


def test_facet_iterators():
    "Iterate over facets"
    mesh = UnitCubeMesh(MPI.comm_world, 5, 5, 5)
    n = 0
    for f in Facets(mesh):
        n += 1
    assert n == mesh.num_facets()


def test_cell_iterators():
    "Iterate over cells"
    mesh = UnitCubeMesh(MPI.comm_world, 5, 5, 5)
    for i in range(4):
        mesh.init(3, i)

    # Test connectivity
    cons = [(i, mesh.topology.connectivity(3, i)) for i in range(4)]

    # Test writability
    for i, con in cons:
        def assign(con, i):
            con(i)[0] = 1
        with pytest.raises(Exception):
            assign(con, i)

    n = 0
    for i, c in enumerate(Cells(mesh)):
        n += 1
        for j, con in cons:
            assert numpy.all(con(i) == c.entities(j))

    assert n == mesh.num_cells()


def test_mixed_iterators():
    "Iterate over vertices of cells"

    mesh = UnitCubeMesh(MPI.comm_world, 5, 5, 5)
    n = 0
    for c in Cells(mesh):
        for v in VertexRange(c):
            n += 1
    assert n == 4 * mesh.num_cells()
