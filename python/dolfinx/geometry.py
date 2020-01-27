# Copyright (C) 2018 Michal Habera and Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfinx import cpp


class BoundingBoxTree:
    def __init__(self, obj, dim=None):
        self._cpp_object = cpp.geometry.BoundingBoxTree(obj, dim)

    @classmethod
    def create_midpoint_tree(cls, mesh):
        """Create a BoundingBoxTree using cell midpoints"""
        tree = cls.__new__(cls)
        tree._cpp_object = cpp.geometry.create_midpoint_tree(mesh)
        return tree

    def str(self):
        """Print for debugging"""
        return self._cpp_object.str()


def compute_first_collision(tree: BoundingBoxTree, x):
    """Compute first collision with the points"""
    return cpp.geometry.compute_first_collision(tree._cpp_object, x)


def compute_first_entity_collision(tree: BoundingBoxTree, mesh, x):
    """Compute fist collision between entities of mesh and the point"""
    return cpp.geometry.compute_first_entity_collision(tree._cpp_object, mesh, x)


def compute_closest_entity(tree: BoundingBoxTree, tree_midpoint, mesh, x):
    """Compute closest entity of the mesh to the point"""
    return cpp.geometry.compute_closest_entity(tree._cpp_object, tree_midpoint._cpp_object, mesh, x)


def compute_collisions_point(tree: BoundingBoxTree, x):
    """Compute collisions with the point"""
    return cpp.geometry.compute_collisions_point(tree._cpp_object, x)


def compute_collisions_bb(tree0: BoundingBoxTree, tree1: BoundingBoxTree):
    """Compute collisions with the bounding box"""
    return cpp.geometry.compute_collisions(tree0._cpp_object, tree1._cpp_object)


def compute_entity_collisions_mesh(tree: BoundingBoxTree, mesh, x):
    """Compute collisions between the point and entities of the mesh"""
    return cpp.geometry.compute_entity_collisions_mesh(tree._cpp_object, mesh, x)


def compute_entity_collisions_bb(tree0: BoundingBoxTree, mesh0, tree1: BoundingBoxTree, mesh1):
    """Compute collisions between the bounding box and entities of meshes"""
    return cpp.geometry.compute_entity_collisions_bb(tree0._cpp_object, tree1._cpp_object, mesh0, mesh1)
