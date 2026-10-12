# Copyright (C) 2026 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tests for the DOLFINx/VTK and DOLFINx/Gmsh cell node ordering maps.

The expected orderings are transcribed from the VTK and Gmsh
specifications rather than derived from the maps under test, so that a
change in behaviour is caught rather than confirmed.
"""

import numpy as np
import pytest

import basix
from dolfinx.io.utils import cell_perm_gmsh, cell_perm_vtk
from dolfinx.mesh import CellType

_basix_cell = {
    CellType.interval: basix.CellType.interval,
    CellType.triangle: basix.CellType.triangle,
    CellType.quadrilateral: basix.CellType.quadrilateral,
    CellType.tetrahedron: basix.CellType.tetrahedron,
    CellType.hexahedron: basix.CellType.hexahedron,
    CellType.prism: basix.CellType.prism,
    CellType.pyramid: basix.CellType.pyramid,
}

# Vertex of the external format that each DOLFINx vertex maps to. VTK and
# Gmsh agree on this for every cell type below. The two differ from
# DOLFINx only on quadrilateral faces, which DOLFINx numbers
# lexicographically and both external formats traverse.
_vertex_perm = {
    CellType.interval: [0, 1],
    CellType.triangle: [0, 1, 2],
    CellType.quadrilateral: [0, 1, 3, 2],
    CellType.tetrahedron: [0, 1, 2, 3],
    CellType.hexahedron: [0, 1, 3, 2, 4, 5, 7, 6],
    CellType.prism: [0, 1, 2, 3, 4, 5],
    CellType.pyramid: [0, 1, 3, 2, 4],
}

# Node ordering of each external format, given for each node as the cell
# vertices it is the centroid of, in that format's vertex numbering. A
# node of a degree <= 2 cell is determined uniquely this way, so these
# tables do not depend on the reference cell coordinates, which the
# formats do not all share.
#
# VTK: the cell classes behind the types listed in
# https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
_vtk_nodes = {
    (CellType.interval, 2): [(0,), (1,)],
    (CellType.interval, 3): [(0,), (1,), (0, 1)],
    (CellType.triangle, 3): [(0,), (1,), (2,)],
    (CellType.triangle, 6): [(0,), (1,), (2,), (0, 1), (1, 2), (2, 0)],
    (CellType.quadrilateral, 4): [(0,), (1,), (2,), (3,)],
    # 8 nodes is serendipity: no interior node.
    (CellType.quadrilateral, 8): [(i,) for i in range(4)] + [(0, 1), (1, 2), (2, 3), (3, 0)],
    (CellType.quadrilateral, 9): [(i,) for i in range(4)]
    + [(0, 1), (1, 2), (2, 3), (3, 0), (0, 1, 2, 3)],
    (CellType.tetrahedron, 4): [(i,) for i in range(4)],
    (CellType.tetrahedron, 10): [(i,) for i in range(4)]
    + [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)],
    (CellType.hexahedron, 8): [(i,) for i in range(8)],
    # 20 nodes is serendipity: no face or interior nodes.
    (CellType.hexahedron, 20): [(i,) for i in range(8)]
    + [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ],
    (CellType.hexahedron, 27): [(i,) for i in range(8)]
    + [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
        (0, 4, 7, 3),
        (1, 2, 6, 5),
        (0, 1, 5, 4),
        (3, 7, 6, 2),
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        tuple(range(8)),
    ],
    (CellType.prism, 6): [(i,) for i in range(6)],
    (CellType.prism, 15): [(i,) for i in range(6)]
    + [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)],
    (CellType.prism, 18): [(i,) for i in range(6)]
    + [
        (0, 1),
        (1, 2),
        (2, 0),
        (3, 4),
        (4, 5),
        (5, 3),
        (0, 3),
        (1, 4),
        (2, 5),
        (0, 1, 4, 3),
        (1, 2, 5, 4),
        (2, 0, 3, 5),
    ],
    (CellType.pyramid, 5): [(i,) for i in range(5)],
    (CellType.pyramid, 13): [(i,) for i in range(5)]
    + [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)],
    (CellType.pyramid, 14): [(i,) for i in range(5)]
    + [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4), (0, 1, 2, 3)],
}

# Gmsh: the node ordering section of the Gmsh manual,
# https://gmsh.info/doc/texinfo/gmsh.html#Node-ordering
_gmsh_nodes = {
    (CellType.interval, 2): [(0,), (1,)],
    (CellType.interval, 3): [(0,), (1,), (0, 1)],
    (CellType.triangle, 3): [(0,), (1,), (2,)],
    (CellType.triangle, 6): [(0,), (1,), (2,), (0, 1), (1, 2), (2, 0)],
    (CellType.quadrilateral, 4): [(0,), (1,), (2,), (3,)],
    (CellType.quadrilateral, 9): [(i,) for i in range(4)]
    + [(0, 1), (1, 2), (2, 3), (3, 0), (0, 1, 2, 3)],
    (CellType.tetrahedron, 4): [(i,) for i in range(4)],
    (CellType.tetrahedron, 10): [(i,) for i in range(4)]
    + [(0, 1), (1, 2), (0, 2), (0, 3), (2, 3), (1, 3)],
    (CellType.hexahedron, 8): [(i,) for i in range(8)],
    (CellType.hexahedron, 27): [(i,) for i in range(8)]
    + [
        (0, 1),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 7),
        (5, 6),
        (6, 7),
        (0, 1, 2, 3),
        (0, 1, 5, 4),
        (0, 3, 7, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (4, 5, 6, 7),
        tuple(range(8)),
    ],
    (CellType.prism, 6): [(i,) for i in range(6)],
    (CellType.prism, 15): [(i,) for i in range(6)]
    + [(0, 1), (0, 2), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (3, 5), (4, 5)],
    (CellType.prism, 18): [(i,) for i in range(6)]
    + [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 4),
        (2, 5),
        (3, 4),
        (3, 5),
        (4, 5),
        (0, 1, 4, 3),
        (0, 2, 5, 3),
        (1, 2, 5, 4),
    ],
    (CellType.pyramid, 5): [(i,) for i in range(5)],
    (CellType.pyramid, 13): [(i,) for i in range(5)]
    + [(0, 1), (0, 3), (0, 4), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4)],
    (CellType.pyramid, 14): [(i,) for i in range(5)]
    + [(0, 1), (0, 3), (0, 4), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4), (0, 1, 2, 3)],
}

# Node counts each map accepts, including the higher degrees that the
# centroid tables above cannot describe.
_vtk_layouts = (
    [(CellType.interval, n) for n in (2, 3, 4, 5)]
    + [(CellType.triangle, n) for n in (3, 6, 10, 15)]
    + [(CellType.quadrilateral, n) for n in (4, 8, 9, 16, 25)]
    + [(CellType.tetrahedron, n) for n in (4, 10, 20, 35)]
    + [(CellType.hexahedron, n) for n in (8, 20, 27, 64)]
    + [(CellType.prism, n) for n in (6, 15, 18)]
    + [(CellType.pyramid, n) for n in (5, 13, 14)]
)

_gmsh_layouts = (
    [(CellType.interval, n) for n in (2, 3, 4, 5)]
    + [(CellType.triangle, n) for n in (3, 6, 10)]
    + [(CellType.quadrilateral, n) for n in (4, 9, 16)]
    + [(CellType.tetrahedron, n) for n in (4, 10, 20)]
    + [(CellType.hexahedron, n) for n in (8, 27)]
    + [(CellType.prism, n) for n in (6, 15, 18)]
    + [(CellType.pyramid, n) for n in (5, 13, 14)]
)


def _dolfinx_nodes(cell_type, num_nodes):
    """DOLFINx node ordering, as the vertex centroid of each node.

    A degree <= 2 Lagrange element places one node on each vertex, then
    each edge, then each quadrilateral face, then (tensor-product cells
    only) the cell interior, in Basix reference cell entity order.
    Serendipity layouts stop early, so entity groups are added only until
    ``num_nodes`` is reached.
    """
    topology = basix.topology(_basix_cell[cell_type])
    nodes = [tuple(v) for v in topology[0]]
    if len(nodes) == num_nodes:
        return nodes

    nodes += [tuple(e) for e in topology[1]]
    if len(nodes) == num_nodes:
        return nodes

    if len(topology) > 3:
        nodes += [tuple(f) for f in topology[2] if len(f) == 4]
        if len(nodes) == num_nodes:
            return nodes

    nodes.append(tuple(range(len(topology[0]))))
    if len(nodes) != num_nodes:
        raise ValueError(f"No degree <= 2 {cell_type} layout with {num_nodes} nodes.")
    return nodes


def _expected_perm(cell_type, num_nodes, external_nodes):
    """Permutation implied by an external format's node ordering.

    Returns ``p`` with ``a_dolfinx[i] = a_external[p[i]]``, by matching
    each DOLFINx node against the external node on the same sub-entity of
    the cell.
    """
    # External vertex number -> DOLFINx vertex number.
    inverse = np.argsort(_vertex_perm[cell_type])
    external = [tuple(sorted(int(inverse[v]) for v in node)) for node in external_nodes]
    assert len(external) == num_nodes
    return [external.index(tuple(sorted(node))) for node in _dolfinx_nodes(cell_type, num_nodes)]


@pytest.mark.parametrize("cell_type, num_nodes", _vtk_layouts)
def test_perm_vtk_is_permutation(cell_type, num_nodes) -> None:
    """Every accepted layout is a permutation that fixes the vertex block."""
    p = cell_perm_vtk(cell_type, num_nodes)
    assert sorted(p) == list(range(num_nodes))
    num_vertices = len(_vertex_perm[cell_type])
    assert sorted(p[:num_vertices]) == list(range(num_vertices))


@pytest.mark.parametrize("cell_type, num_nodes", _gmsh_layouts)
def test_perm_gmsh_is_permutation(cell_type, num_nodes) -> None:
    p = cell_perm_gmsh(cell_type, num_nodes)
    assert sorted(p) == list(range(num_nodes))
    num_vertices = len(_vertex_perm[cell_type])
    assert sorted(p[:num_vertices]) == list(range(num_vertices))


@pytest.mark.parametrize("cell_type, vertex_perm", _vertex_perm.items())
def test_vertex_perm(cell_type, vertex_perm) -> None:
    """A linear cell permutes to the external vertex ordering."""
    assert list(cell_perm_vtk(cell_type, len(vertex_perm))) == vertex_perm
    assert list(cell_perm_gmsh(cell_type, len(vertex_perm))) == vertex_perm


@pytest.mark.parametrize("cell_type, num_nodes", _vtk_nodes.keys())
def test_perm_vtk(cell_type, num_nodes) -> None:
    """Each node maps to the VTK node on the same sub-entity."""
    p = _expected_perm(cell_type, num_nodes, _vtk_nodes[cell_type, num_nodes])
    assert list(cell_perm_vtk(cell_type, num_nodes)) == p


@pytest.mark.parametrize("cell_type, num_nodes", _gmsh_nodes.keys())
def test_perm_gmsh(cell_type, num_nodes) -> None:
    """Each node maps to the Gmsh node on the same sub-entity."""
    p = _expected_perm(cell_type, num_nodes, _gmsh_nodes[cell_type, num_nodes])
    assert list(cell_perm_gmsh(cell_type, num_nodes)) == p


@pytest.mark.parametrize("cell_type, num_nodes", _vtk_nodes.keys())
def test_dolfinx_nodes(cell_type, num_nodes) -> None:
    """Check the DOLFINx node ordering assumed above against Basix.

    The sub-entity that `_dolfinx_nodes` assigns to each node must hold
    the node, i.e. the Basix node coordinate is the centroid of that
    sub-entity's vertices. Serendipity layouts have no Basix element and
    are skipped.
    """
    cell = _basix_cell[cell_type]
    degree = 1 if num_nodes == len(_vertex_perm[cell_type]) else 2
    e = basix.create_element(basix.ElementFamily.P, cell, degree, basix.LagrangeVariant.equispaced)
    if e.dim != num_nodes:
        pytest.skip(f"No Basix element with {num_nodes} nodes")

    x = basix.geometry(cell)
    for point, node in zip(e.points, _dolfinx_nodes(cell_type, num_nodes), strict=True):
        assert np.allclose(point, x[list(node)].mean(axis=0))
