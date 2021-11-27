// Copyright (C) 2010 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <vector>
#include <xtl/xspan.hpp>

namespace dolfinx::mesh
{
class Mesh;
} // namespace dolfinx::mesh

namespace dolfinx::refinement
{

/// Compute incident edges for a set of mesh entities
/// @param[in] mesh The mesh
/// @param[in] entities Mesh entity indices
/// @param[in] dim Topological dimension of the entities
/// @return Edges (indicies) that are incident to @p entities
std::vector<std::int32_t>
compute_marked_edges(const mesh::Mesh& mesh,
                     const xtl::span<const std::int32_t> entities, int dim);

/// Create uniformly refined mesh
///
/// @param[in] mesh The mesh from which to build a refined Mesh
/// @param[in] redistribute Optional argument to redistribute the
/// refined mesh if mesh is a distributed mesh.
/// @return A refined mesh
mesh::Mesh refine(const mesh::Mesh& mesh, bool redistribute = true);

/// Create locally refined mesh
///
/// @param[in] mesh The mesh from which to build a refined Mesh
/// @param[in] entity_markers MeshTags listing which mesh entity indices
/// should be split by this refinement. The values are ignored.
/// @param[in] redistribute Optional argument to redistribute the
/// refined mesh if mesh is a distributed mesh.
/// @return A locally refined mesh
mesh::Mesh refine(const mesh::Mesh& mesh,
                  const xtl::span<const std::int32_t>& edges,
                  bool redistribute = true);

} // namespace dolfinx::refinement
