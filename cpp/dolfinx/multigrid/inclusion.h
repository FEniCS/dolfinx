// Copyright (C) 2025 Paul T. Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <concepts>
#include <cstdint>
#include <vector>

#include "dolfinx/mesh/Mesh.h"

namespace dolfinx::multigrid
{

/// @brief Computes an inclusion map: a map between vertex indices from one mesh
/// to another.
///
/// @param mesh_from Domain of the map
/// @param mesh_to Range of the map
///
/// @return Inclusion map, the `i`-th component is the vertex index of the
/// vertex with the same coordinates in `mesh_to` and `-1` if it can not be
/// found (locally!) in `mesh_to`. If `map[i] != -1` it holds
/// `mesh_from.geometry.x()[i:i+3] == mesh_to.geometry.x()[map[i]:map[i]+3]`.
///
/// @note Invoking `inclusion_map` on a `(mesh_coarse, mesh_fine)` tuple, where
/// `mesh_fine` is produced by refinement with
/// `IdentityPartitionerPlaceholder()` option, the returned `map` is guaranteed
/// to match all vertices for all locally owned vertices (not for the ghost
/// vertices).
///
template <std::floating_point T>
std::vector<std::int32_t>
inclusion_mapping(const dolfinx::mesh::Mesh<T>& mesh_from,
                  const dolfinx::mesh::Mesh<T>& mesh_to);

} // namespace dolfinx::multigrid
