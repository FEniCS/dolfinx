// Copyright (C) 2020 Chris Richardson
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>

namespace dolfinx
{
namespace geometry
{

/// Calculate the distance between two convex bodies p and q, each defined by a
/// set of points, using the Gilbert–Johnson–Keerthi (GJK) distance algorithm.
/// @param[in] p Body 1 list of points
/// @param[in] q Body 2 list of points
/// @return shortest vector between bodies
Eigen::Vector3d
compute_distance_gjk(const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& p,
           const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& q);
} // namespace geometry
} // namespace dolfinx