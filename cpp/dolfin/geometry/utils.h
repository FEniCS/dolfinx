// Copyright (C) 2019 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>

namespace dolfin
{
namespace mesh
{
class Cell;
}

namespace geometry
{
/// Compute squared distance from a given point to the nearest point on
/// a cell (only simplex cells are supported at this stage)
double squared_distance(const mesh::Cell& cell, const Eigen::Vector3d& point);

/// Compute squared distance to given point. This version takes
/// the three vertex coordinates as 3D points. This makes it
/// possible to reuse this function for computing the (squared)
/// distance to a tetrahedron.
double squared_distance_triangle(const Eigen::Vector3d& point,
                                 const Eigen::Vector3d& a,
                                 const Eigen::Vector3d& b,
                                 const Eigen::Vector3d& c);

/// Compute squared distance to given point. This version takes
/// the two vertex coordinates as 3D points. This makes it
/// possible to reuse this function for computing the (squared)
/// distance to a triangle.
double squared_distance_interval(const Eigen::Vector3d& point,
                                 const Eigen::Vector3d& a,
                                 const Eigen::Vector3d& b);

} // namespace geometry
} // namespace dolfin