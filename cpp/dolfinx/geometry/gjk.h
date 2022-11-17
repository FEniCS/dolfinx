// Copyright (C) 2020 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <span>

namespace dolfinx::geometry
{

/// Calculate the distance between two convex bodies p and q, each
/// defined by a set of points, using the Gilbert–Johnson–Keerthi (GJK)
/// distance algorithm.
///
/// @param[in] p Body 1 list of points, shape (num_points, 3). Row-major
/// storage.
/// @param[in] q Body 2 list of points, shape (num_points, 3). Row-major
/// storage.
/// @return shortest vector between bodies
std::array<double, 3> compute_distance_gjk(std::span<const double> p,
                                           std::span<const double> q);

} // namespace dolfinx::geometry
