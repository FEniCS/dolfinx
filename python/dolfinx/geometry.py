# Copyright (C) 2018 Michal Habera and Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfinx import cpp


class BoundingBoxTree:
    def __init__(self, obj, dim=None, entities=None):
        if entities is None:
            self._cpp_object = cpp.geometry.BoundingBoxTree(obj, dim)
        else:
            self._cpp_object = cpp.geometry.BoundingBoxTree(obj, dim, entities)

    def num_bboxes(self):
        return self._cpp_object.num_bboxes()

    @classmethod
    def create_midpoint_tree(cls, mesh):
        """Create a BoundingBoxTree using cell midpoints"""
        tree = cls.__new__(cls)
        tree._cpp_object = cpp.geometry.create_midpoint_tree(mesh)
        return tree

    def str(self):
        """Print for debugging"""
        return self._cpp_object.str()

    def compute_global_tree(self, comm):
        """Create a global BoundingBoxTree for per-process bounding boxes for a MPI communicator"""
        tree = BoundingBoxTree.__new__(BoundingBoxTree)
        tree._cpp_object = self._cpp_object.compute_global_tree(comm)
        return tree


def compute_closest_entity(tree: BoundingBoxTree, tree_midpoint, mesh, x):
    """Compute closest entity of the mesh to the point"""
    return cpp.geometry.compute_closest_entity(tree._cpp_object, tree_midpoint._cpp_object, mesh, x)


def compute_collisions_point(tree: BoundingBoxTree, x):
    """Compute collisions with the point"""
    return cpp.geometry.compute_collisions_point(tree._cpp_object, x)


def compute_colliding_cells(tree: BoundingBoxTree, mesh, x, n=1):
    """Return cells which the point x lies within"""
    candidate_cells = cpp.geometry.compute_collisions_point(tree._cpp_object, x)
    return cpp.geometry.select_colliding_cells(mesh, candidate_cells, x, n)


def compute_collisions(tree0: BoundingBoxTree, tree1: BoundingBoxTree):
    """Compute collisions with the bounding box"""
    return cpp.geometry.compute_collisions(tree0._cpp_object, tree1._cpp_object)
