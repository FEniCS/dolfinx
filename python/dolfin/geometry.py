# -*- coding: utf-8 -*-
# Copyright (C) 2018 Michal Habera
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import cpp


def compute_first_collision(tree, x):
    """Compute first collision with the points"""
    return cpp.geometry.compute_first_collision(tree._cpp_object, x)


def compute_first_entity_collision(tree, mesh, x):
    """Compute fist collision between entities of mesh and the point"""
    return cpp.geometry.compute_first_entity_collision(tree._cpp_object, mesh, x)


def compute_closest_entity(tree, tree_midpoint, mesh, x):
    """Compute closest entity of the mesh to the point"""
    return cpp.geometry.compute_closest_entity(tree._cpp_object, tree_midpoint._cpp_object, mesh, x)


def compute_collisions_point(tree, x):
    """Compute collisions with the point"""
    return cpp.geometry.compute_collisions_point(tree._cpp_object, x)


def create_midpoint_tree(mesh):
    return BoundingBoxTree(cpp.geometry.create_midpoint_tree(mesh))


class BoundingBoxTree:
    def __init__(self, obj, dim=None):
        """Create bounding box tree"""
        if isinstance(obj, cpp.geometry.BoundingBoxTree):
            self._cpp_object = obj
        else:
            self._cpp_object = cpp.geometry.BoundingBoxTree(obj, dim)

    def compute_collisions_bb(self, bb: "BoundingBoxTree"):
        """Compute collisions with the bounding box"""
        return self._cpp_object.compute_collisions(bb._cpp_object)

    def compute_entity_collisions_mesh(self, point, mesh):
        """Compute collisions between the point and entities of the mesh"""
        return self._cpp_object.compute_entity_collisions(
            point, mesh)

    def compute_entity_collisions_bb_mesh(self, bb: "BoundingBoxTree", mesh1,
                                          mesh2):
        """Compute collisions between the bounding box and entities of meshes"""
        return self._cpp_object.compute_entity_collisions(
            bb._cpp_object, mesh1, mesh2)

    def str(self):
        """Print for debugging"""
        return self._cpp_object.str()
