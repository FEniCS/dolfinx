# -*- coding: utf-8 -*-
# Copyright (C) 2018 Michal Habera
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import cpp


class BoundingBoxTree:
    def __init__(self, gdim=None):
        """Initialise from geometric dimension"""
        if gdim is not None:
            self._cpp_object = cpp.geometry.BoundingBoxTree(gdim)

    # def fromcpp(self, cpp_object):
    #     self._cpp_object = cpp_object

    def build_points(self, points: list):
        """Build from cloud of points"""
        self._cpp_object.build(points)

    def build_mesh(self, mesh, tdim: int):
        """Build from mesh entities of given topological dimension"""
        self._cpp_object.build(mesh, tdim)

    def compute_collisions_points(self, point):
        """Compute collisions with the point"""
        return self._cpp_object.compute_collisions(point)

    def compute_collisions_bb(self, bb):
        """Compute collisions with the bounding box"""
        return self._cpp_object.compute_collisions(bb)

    def compute_entity_collisions_mesh(self, point, mesh):
        """Compute collisions between the point and entities of the mesh"""
        return self._cpp_object.compute_entity_collisions(point, mesh)

    def compute_entity_collisions_bb_mesh(self, bb, mesh1, mesh2):
        """Compute collisions between the bounding box and entities of meshes"""
        return self._cpp_object.compute_entity_collisions(bb, mesh1, mesh2)

    def compute_first_collision(self, point):
        """Compute first collision with the point"""
        return self._cpp_object.compute_first_collision(point)

    def compute_first_entity_collision(self, point, mesh):
        """Compute fist collision between entities of mesh and the point"""
        return self._cpp_object.compute_first_entity_collision(point, mesh)

    def compute_closest_entity(self, point, mesh):
        """Compute closest entity of the mesh to the point"""
        return self._cpp_object.compute_closest_entity(point, mesh)

    def str(self, point, mesh):
        """Print for debbuging"""
        return self._cpp_object.str()


class Point:
    """Point represents a point in 3D euclidean space"""

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """Initialise from coordinates"""
        self._cpp_object = cpp.geometry.Point(x, y, z)

    # def fromcpp(self, cpp_object=None):
    #     self._cpp_object = cpp_object

    def __getitem__(self, key):
        return self._cpp_object[key]

    def __setitem__(self, index, value):
        self._cpp_object[index] = value

    def __add__(self, other: Point):
        return self._cpp_object.__add__(other._cpp_object)

    def __sub__(self, other: Point):
        return self._cpp_object.__sub__(other._cpp_object)

    def __eq__(self, other: Point):
        return self._cpp_object.__eq__(other._cpp_object)

    def __mul__(self, other: Point):
        return self._cpp_object.__mul__(other._cpp_object)

    def __div__(self, other: Point):
        return self._cpp_object.__div__(other._cpp_object)

    def array(self):
        """Return as array"""
        return self._cpp_object.array()

    def norm(self):
        """Compute euclidean norm of a vector from origin to the Point"""
        return self._cpp_object.norm()

    def distance(self, other: Point):
        """Compute euclidean distance to the point"""
        return self._cpp_object.distance(other._cpp_object)
