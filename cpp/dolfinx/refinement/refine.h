// Copyright (C) 2010 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>

namespace dolfinx
{

namespace mesh
{
// Forward declarations
class Mesh;
template <typename T>
class MeshTags;
} // namespace mesh

namespace refinement
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
/// @param[in] cell_markers A mesh function over integers specifying
///     which cells should be refined (value == 1) (and which should not
///     (any other integer value)).
/// @param[in] redistribute Optional argument to redistribute the
///     refined mesh if mesh is a distributed mesh.
/// @return A locally refined mesh
mesh::Mesh refine(const mesh::Mesh& mesh,
                  const mesh::MeshTags<std::int8_t>& cell_markers,
                  bool redistribute = true);

} // namespace refinement
} // namespace dolfinx
