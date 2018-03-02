// Copyright (C) 2014-2018 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include<cstdint>
#include<vector>

#pragma once

namespace dolfin
{

namespace mesh
{
class Mesh;
}

namespace refinement
{
class ParallelRefinement;

/// Implementation of the refinement method described in
/// Plaza and Carey "Local refinement of simplicial grids
/// based on the skeleton"
/// (Applied Numerical Mathematics 32 (2000) 195-218)
///
class PlazaRefinementND
{
public:
  /// Uniform refine, optionally redistributing and
  /// optionally calculating the parent-child relation for facets (in 2D)
  ///
  ///  @param new_mesh
  ///     New Mesh
  ///  @param mesh
  ///     Input mesh to be refined
  ///  @param redistribute
  ///     Flag to call the Mesh Partitioner to redistribute after refinement
  ///
  static void refine(mesh::Mesh& new_mesh, const mesh::Mesh& mesh, bool redistribute);

  /// Refine with markers, optionally redistributing
  ///
  /// @param new_mesh
  ///    New Mesh
  /// @param mesh
  ///    Input mesh to be refined
  /// @param refinement_marker
  ///    MeshFunction listing MeshEntities which should be split by this
  ///    refinement
  /// @param redistribute
  ///     Flag to call the Mesh Partitioner to redistribute after refinement
  ///
  static void refine(mesh::Mesh& new_mesh, const mesh::Mesh& mesh,
                     const mesh::MeshFunction<bool>& refinement_marker,
                     bool redistribute);

  /// Refine with markers, optionally calculating facet relations, and
  /// saving relation data in MeshRelation structure
  /// @param new_mesh
  ///    New Mesh
  /// @param mesh
  ///    Input mesh to be refined
  /// @param refinement_marker
  ///    MeshFunction listing MeshEntities which should be split by this
  ///    refinement
  ///
  static void refine(mesh::Mesh& new_mesh, const mesh::Mesh& mesh,
                     const mesh::MeshFunction<bool>& refinement_marker);

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
  static void get_simplices(std::vector<std::size_t>& simplex_set,
                            const std::vector<bool>& marked_edges,
                            const std::vector<std::int32_t>& longest_edge,
                            std::size_t tdim, bool uniform);

private:
  // Get the longest edge of each face (using local mesh index)
  static void face_long_edge(std::vector<std::int32_t>& long_edge,
                             std::vector<bool>& edge_ratio_ok,
                             const mesh::Mesh& mesh);

  // 2D version of subdivision allowing for uniform subdivision (flag)
  static void get_triangles(std::vector<std::size_t>& tri_set,
                            const std::vector<bool>& marked_edges,
                            const std::int32_t longest_edge, bool uniform);

  // 3D version of subdivision
  static void get_tetrahedra(std::vector<std::size_t>& tet_set,
                             const std::vector<bool>& marked_edges,
                             const std::vector<std::int32_t>& longest_edge);

  // Convenient interface for both uniform and marker refinement
  static void do_refine(mesh::Mesh& new_mesh, const mesh::Mesh& mesh,
                        ParallelRefinement& p_ref,
                        const std::vector<std::int32_t>& long_edge,
                        const std::vector<bool>& edge_ratio_ok,
                        bool redistribute);

  // Propagate edge markers according to rules (longest edge
  // of each face must be marked, if any edge of face is marked)
  static void enforce_rules(ParallelRefinement& p_ref, const mesh::Mesh& mesh,
                            const std::vector<std::int32_t>& long_edge);
};
}
}
