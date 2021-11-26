// Copyright (C) 2010 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>

namespace dolfinx::mesh
{
class Mesh;
template <typename T>
class MeshTags;
} // namespace dolfinx::mesh

namespace dolfinx::refinement
{

/// Create uniformly refined mesh
///
/// @param[in] mesh The mesh from which to build a refined Mesh
/// @param[in] redistribute Optional argument to redistribute the
///     refined mesh if mesh is a distributed mesh.
/// @return A refined mesh
mesh::Mesh refine(const mesh::Mesh& mesh, bool redistribute = true);

/// Create locally refined mesh
///
/// @param[in] mesh The mesh from which to build a refined Mesh
/// @param[in] entity_markers MeshTags listing which mesh entity indices
/// should be split by this refinement. The values are ignored.
/// @param[in] redistribute Optional argument to redistribute the
///     refined mesh if mesh is a distributed mesh.
/// @return A locally refined mesh
mesh::Mesh refine(const mesh::Mesh& mesh,
                  const mesh::MeshTags<std::int8_t>& entity_markers,
                  bool redistribute = true);

} // namespace dolfinx::refinement
