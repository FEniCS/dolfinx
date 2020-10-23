// Copyright (C) 2014-2018 Chris Richardson
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <cstdint>
#include <utility>
#include <vector>

#pragma once

namespace dolfinx
{

namespace mesh
{
class Mesh;
template <typename T>
class MeshTags;
} // namespace mesh

namespace refinement
{
class ParallelRefinement;

/// Implementation of the refinement method described in Plaza and Carey
/// "Local refinement of simplicial grids based on the skeleton"
/// (Applied Numerical Mathematics 32 (2000) 195-218)

namespace PlazaRefinementND
{

/// Uniform refine, optionally redistributing and optionally
/// calculating the parent-child relation for facets (in 2D)
///
/// @param[in] mesh Input mesh to be refined
/// @param[in] redistribute Flag to call the Mesh Partitioner to
///   redistribute after refinement
/// @return New mesh
mesh::Mesh refine(const mesh::Mesh& mesh, bool redistribute);

/// Refine with markers, optionally redistributing
///
/// @param[in] mesh Input mesh to be refined
/// @param[in] refinement_marker MeshTags listing mesh entities which
///   should be split by this refinement. Value == 1 means "refine",
///   any other value means "do not refine".
/// @param[in] redistribute Flag to call the Mesh Partitioner to
///   redistribute after refinement
/// @return New Mesh
mesh::Mesh refine(const mesh::Mesh& mesh,
                  const mesh::MeshTags<std::int8_t>& refinement_marker,
                  bool redistribute);

} // namespace PlazaRefinementND
} // namespace refinement
} // namespace dolfinx
