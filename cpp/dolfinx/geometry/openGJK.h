// Copyright (C) 2020 Mattia Montanari, Chris Richardson
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

/// Calculate the distance between two convex bodies bd1 and bd2
/// @param[in] bd1 Body 1 list of vertices
/// @param[in] bd2 Body 2 list of vertices
/// @return shortest vector between bodies
Eigen::Vector3d gjk_vector(
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& bd1,
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& bd2);
} // namespace geometry
} // namespace dolfinx