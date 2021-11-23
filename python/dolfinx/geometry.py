# Copyright (C) 2018-2020 Michal Habera and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfinx import cpp as _cpp
from dolfinx.cpp.geometry import (create_midpoint_tree, compute_closest_entity,  # noqa
                                  compute_collisions, compute_distance_gjk,
                                  squared_distance)


class BoundingBoxTree(_cpp.geometry.BoundingBoxTree):
    def __init__(self, mesh, dim):
        try:
            super().__init__(mesh, dim)
        except TypeError:
            super().__init__(mesh._cpp_object, dim)


def compute_colliding_cells(mesh, candidates, x):
    try:
        return _cpp.geometry.compute_colliding_cells(mesh, candidates, x)
    except TypeError:
        return _cpp.geometry.compute_colliding_cells(mesh._cpp_object, candidates, x)
