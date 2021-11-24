# Copyright (C) 2018-2020 Michal Habera and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfinx
from dolfinx import cpp as _cpp
from dolfinx.cpp.geometry import (create_midpoint_tree, compute_closest_entity,  # noqa
                                  compute_collisions, compute_distance_gjk)


class BoundingBoxTree(_cpp.geometry.BoundingBoxTree):
    def __init__(self, mesh: dolfinx.mesh.Mesh, dim: int, entities=None, padding=0.0):
        map = mesh.topology.index_map(dim)
        if map is None:
            raise RuntimeError(f"Mesh entities of dimension {dim} have not been created.")
        if entities is None:
            entities = range(0, map.size_local + map.num_ghosts)

        super().__init__(mesh._cpp_object, dim, entities, padding)


def compute_colliding_cells(mesh, candidates, x):
    return _cpp.geometry.compute_colliding_cells(mesh._cpp_object, candidates, x)


def squared_distance(mesh, dim, entities, points):
    return _cpp.geometry.squared_distance(mesh._cpp_object, dim, entities, points)
