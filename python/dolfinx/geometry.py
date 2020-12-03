# Copyright (C) 2018-2020 Michal Habera and Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfinx.cpp.geometry import (BoundingBoxTree, create_midpoint_tree, compute_closest_entity, # noqa
                                  compute_collisions_point, compute_collisions, compute_distance_gjk,
                                  squared_distance, select_colliding_cells)
