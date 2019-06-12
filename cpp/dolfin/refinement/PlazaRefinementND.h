// Copyright (C) 2014-2018 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <cstdint>
#include <utility>
#include <vector>

#pragma once

namespace dolfin
{

namespace mesh
{
class Mesh;
template <typename T>
class MeshFunction;
} // namespace mesh

namespace refinement
{
class ParallelRefinement;

/// Implementation of the refinement method described in Plaza and Carey
/// "Local refinement of simplicial grids based on the skeleton"
/// (Applied Numerical Mathematics 32 (2000) 195-218)
///
class PlazaRefinementND
{
public:
  /// Uniform refine, optionally redistributing and
  /// optionally calculating the parent-child relation for facets (in 2D)
  ///
  ///  @param mesh
  ///     Input mesh to be refined
  ///  @param redistribute
  ///     Flag to call the Mesh Partitioner to redistribute after refinement
  ///  @returns mesh::Mesh
  ///     New mesh
  ///
  static mesh::Mesh refine(const mesh::Mesh& mesh, bool redistribute);

  /// Refine with markers, optionally redistributing
  ///
  /// @param mesh
  ///    Input mesh to be refined
  /// @param refinement_marker
  ///    MeshFunction listing MeshEntities which should be split by this
  ///    refinement
  /// @param redistribute
  ///     Flag to call the Mesh Partitioner to redistribute after refinement
  /// @returns mesh::Mesh
  ///    New Mesh
  ///
  static mesh::Mesh refine(const mesh::Mesh& mesh,
                           const mesh::MeshFunction<bool>& refinement_marker,
                           bool redistribute);

  /// Get the subdivision of an original simplex into smaller
  /// simplices, for a given set of marked edges, and the
  /// longest edge of each facet (cell local indexing).
  /// A flag indicates if a uniform subdivision is preferable in 2D.
  /// @param simplex_set
  ///   Returned set of triangles/tets topological description
  /// @param marked_edges
  ///   Vector indicating which edges are to be split
  /// @param longest_edge
  ///   Vector indicating the longest edge for each triangle. For tdim=2, one
  ///   entry, for tdim=3, four entries.
  /// @param tdim
  ///   Topological dimension (2 or 3)
  /// @param uniform
  ///   Make a "uniform" subdivision with all triangles being similar shape
  ///
  static std::vector<std::int32_t>
  get_simplices(const std::vector<bool>& marked_edges,
                const std::vector<std::int32_t>& longest_edge,
                std::int32_t tdim, bool uniform);
};
} // namespace refinement
} // namespace dolfin
