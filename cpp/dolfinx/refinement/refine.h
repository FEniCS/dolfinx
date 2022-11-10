// Copyright (C) 2010-2021 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace dolfinx::mesh
{
class Mesh;
} // namespace dolfinx::mesh

namespace dolfinx::refinement
{

/// Create a uniformly refined mesh
///
/// @param[in] mesh The mesh from which to build a refined Mesh
/// @param[in] redistribute Optional argument to redistribute the
/// refined mesh if mesh is a distributed mesh.
/// @return A refined mesh
mesh::Mesh refine(const mesh::Mesh& mesh, bool redistribute = true);

/// Create a locally refined mesh
///
/// @param[in] mesh The mesh from which to build a refined Mesh
/// @param[in] edges Indices of the edges that should be split by this
/// refinement. mesh::compute_incident_entities can be used to compute
/// the edges that are incident to other entities, e.g. incident to
/// cells.
/// @param[in] redistribute Optional argument to redistribute the
/// refined mesh if mesh is a distributed mesh.
/// @return A locally refined mesh
mesh::Mesh refine(const mesh::Mesh& mesh, std::span<const std::int32_t> edges,
                  bool redistribute = true);

} // namespace dolfinx::refinement
