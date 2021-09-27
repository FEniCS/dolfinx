// Copyright (C) 2020 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <xtensor/xfixed.hpp>
#include <xtensor/xtensor.hpp>

namespace dolfinx::geometry
{

/// Calculate the distance between two convex bodies p and q, each
/// defined by a set of points, using the Gilbert–Johnson–Keerthi (GJK)
/// distance algorithm.
///
/// @param[in] p Body 1 list of points, shape (num_points, 3)
/// @param[in] q Body 2 list of points, shape (num_points, 3)
/// @return shortest vector between bodies
xt::xtensor_fixed<double, xt::xshape<3>>
compute_distance_gjk(const xt::xtensor<double, 2>& p,
                     const xt::xtensor<double, 2>& q);

} // namespace dolfinx::geometry
