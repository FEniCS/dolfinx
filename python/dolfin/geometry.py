# -*- coding: utf-8 -*-
# Copyright (C) 2018 Michal Habera
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import cpp


class BoundingBoxTree:
    def __init__(self, obj, dim):
        """Create bounding box tree"""
        self._cpp_object = cpp.geometry.BoundingBoxTree(obj, dim)

    def compute_collisions_point(self, point: "Point"):
        """Compute collisions with the point"""
        return self._cpp_object.compute_collisions(point._cpp_object)

    def compute_collisions_bb(self, bb: "BoundingBoxTree"):
        """Compute collisions with the bounding box"""
        return self._cpp_object.compute_collisions(bb._cpp_object)

    def compute_entity_collisions_mesh(self, point: "Point", mesh):
        """Compute collisions between the point and entities of the mesh"""
        return self._cpp_object.compute_entity_collisions(
            point._cpp_object, mesh)

    def compute_entity_collisions_bb_mesh(self, bb: "BoundingBoxTree", mesh1,
                                          mesh2):
        """Compute collisions between the bounding box and entities of meshes"""
        return self._cpp_object.compute_entity_collisions(
            bb._cpp_object, mesh1, mesh2)

    def compute_first_collision(self, point: "Point"):
        """Compute first collision with the point"""
        return self._cpp_object.compute_first_collision(point._cpp_object)

    def compute_first_entity_collision(self, point: "Point", mesh):
        """Compute fist collision between entities of mesh and the point"""
        return self._cpp_object.compute_first_entity_collision(
            point._cpp_object, mesh)

    def compute_closest_entity(self, point: "Point", mesh):
        """Compute closest entity of the mesh to the point"""
        return self._cpp_object.compute_closest_entity(point._cpp_object, mesh)

    def str(self):
        """Print for debugging"""
        return self._cpp_object.str()


class Point:
    """Point represents a point in 3D Euclidean space"""

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """Initialise from coordinates"""
        self._cpp_object = cpp.geometry.Point(x, y, z)

    def __getitem__(self, key):
        return self._cpp_object[key]

    def __setitem__(self, index, value):
        self._cpp_object[index] = value

    def __add__(self, other):
        return self._cpp_object.__add__(getattr(other, "_cpp_object", other))

    def __sub__(self, other):
        return self._cpp_object.__sub__(getattr(other, "_cpp_object", other))

    def __eq__(self, other):
        return self._cpp_object.__eq__(getattr(other, "_cpp_object", other))

    def __mul__(self, other):
        return self._cpp_object.__mul__(getattr(other, "_cpp_object", other))

    def __div__(self, other):
        return self._cpp_object.__div__(getattr(other, "_cpp_object", other))

    def array(self):
        """Return as array"""
        return self._cpp_object.array()

    def norm(self):
        """Compute euclidean norm of a vector from origin to the Point"""
        return self._cpp_object.norm()

    def distance(self, other: "Point"):
        """Compute euclidean distance to the point"""
        return self._cpp_object.distance(getattr(other, "_cpp_object", other))
