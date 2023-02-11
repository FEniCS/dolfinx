// Copyright (C) 2010-2023 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <span>

namespace dolfinx::mesh
{
class Mesh;
} // namespace dolfinx::mesh

namespace dolfinx::refinement
{
/// @brief Create a uniformly refined mesh.
///
/// @param[in] mesh Mesh to create a new, refined mesh from
/// @param[in] redistribute If `true` refined mesh is re-partitioned
/// across MPI ranks.
/// @return Refined mesh
mesh::Mesh refine(const mesh::Mesh& mesh, bool redistribute = true);

/// @brief Create a locally refined mesh.
///
/// @param[in] mesh Mesh to create a new, refined mesh from.
/// @param[in] edges Indices of the edges that should be split during
/// refinement. mesh::compute_incident_entities can be used to compute
/// the edges that are incident to other entities, e.g. incident to
/// cells.
/// @param[in] redistribute If `true` refined mesh is re-partitioned
/// across MPI ranks.
/// @return Refined mesh.
mesh::Mesh refine(const mesh::Mesh& mesh, std::span<const std::int32_t> edges,
                  bool redistribute = true);
} // namespace dolfinx::refinement
