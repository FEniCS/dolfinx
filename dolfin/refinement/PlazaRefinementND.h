// Copyright (C) 2014 Chris Richardson
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//

#ifndef __PLAZA_REFINEMENT_ND_H
#define __PLAZA_REFINEMENT_ND_H

namespace dolfin
{
  class Mesh;
  class MeshRelation;
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
    ///  @param calculate_parent_facets
    ///     Flag to build parent facet information, needed to propagate information
    ///     on boundaries
    static void refine(Mesh& new_mesh, const Mesh& mesh, bool redistribute,
                       bool calculate_parent_facets);

    /// Refine with markers, optionally redistributing
    /// and optionally calculating the parent-child relation for facets (in 2D)
    ///
    /// @param new_mesh
    ///    New Mesh
    /// @param mesh
    ///    Input mesh to be refined
    /// @param refinement_marker
    ///    MeshFunction listing MeshEntities which should be split by this refinement
    /// @param redistribute
    ///     Flag to call the Mesh Partitioner to redistribute after refinement
    /// @param calculate_parent_facets
    ///     Flag to build parent facet information, needed to propagate information
    ///     on boundaries
    static void refine(Mesh& new_mesh, const Mesh& mesh,
                       const MeshFunction<bool>& refinement_marker,
                       bool redistribute,
                       bool calculate_parent_facets);

    /// Refine with markers, optionally calculating facet relations, and
    /// saving relation data in MeshRelation structure
    /// @param new_mesh
    ///    New Mesh
    /// @param mesh
    ///    Input mesh to be refined
    /// @param refinement_marker
    ///    MeshFunction listing MeshEntities which should be split by this refinement
    /// @param calculate_parent_facets
    ///    Flag to build parent facet information, needed to propagate information
    ///    on boundaries
    /// @param mesh_relation
    ///    New relationship between the two meshes
    static void refine(Mesh& new_mesh, const Mesh& mesh,
                       const MeshFunction<bool>& refinement_marker,
                       bool calculate_parent_facets,
                       MeshRelation& mesh_relation);


    /// Get the subdivision of an original simplex into smaller
    /// simplices, for a given set of marked edges, and the
    /// longest edge of each facet (cell local indexing).
    /// A flag indicates if a uniform subdivision is preferable in 2D.
    /// @param simplex_set
    ///   Returned set of triangles/tets topological description
    /// @param marked_edges
    ///   Vector indicating which edges are to be split
    /// @param longest_edge
    ///   Vector indicating the longest edge for each triangle. For tdim=2, one entry,
    ///   for tdim=3, four entries.
    /// @param tdim
    ///   Topological dimension (2 or 3)
    /// @param uniform
    ///   Make a "uniform" subdivision with all triangles being similar shape
    static void get_simplices
      (std::vector<std::size_t>& simplex_set,
       const std::vector<bool>& marked_edges,
       const std::vector<std::size_t>& longest_edge,
       std::size_t tdim, bool uniform);

  private:

    // Get the longest edge of each face (using local mesh index)
    static void face_long_edge(std::vector<unsigned int>& long_edge,
                               std::vector<bool>& edge_ratio_ok,
                               const Mesh& mesh);

    // 2D version of subdivision allowing for uniform subdivision (flag)
    static void get_triangles
      (std::vector<std::size_t>& tri_set,
       const std::vector<bool>& marked_edges,
       const std::size_t longest_edge,
       bool uniform);

    // 3D version of subdivision
    static void get_tetrahedra
      (std::vector<std::size_t>& tet_set,
       const std::vector<bool>& marked_edges,
       const std::vector<std::size_t>& longest_edge);

    // Convenient interface for both uniform and marker refinement
    static void do_refine(Mesh& new_mesh, const Mesh& mesh,
                          ParallelRefinement& p_ref,
                          const std::vector<unsigned int>& long_edge,
                          const std::vector<bool>& edge_ratio_ok,
                          bool redistribute,
                          bool calculate_parent_facets,
                          MeshRelation& mesh_relation);

    // Propagate edge markers according to rules (longest edge
    // of each face must be marked, if any edge of face is marked)
    static void enforce_rules(ParallelRefinement& p_ref,
                              const Mesh& mesh,
                              const std::vector<unsigned int>& long_edge);

    // Add parent facet markers to new mesh, based on new vertices
    // Only works in 2D at present
    static void set_parent_facet_markers(const Mesh& mesh, Mesh& new_mesh,
                  const std::map<std::size_t, std::size_t>& new_vertex_map);


  };

}

#endif
