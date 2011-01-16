// Copyright (C) 2006 Johan Hoffman.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2010.
// Modified by Anders Logg, 2010.
//
// First added:  2006-11-01
// Last changed: 2010-02-26

#ifndef __LOCAL_MESH_REFINEMENT_H
#define __LOCAL_MESH_REFINEMENT_H

#include <vector>
#include "dolfin/common/types.h"

namespace dolfin
{

  // Forward declarations
  class Cell;
  class Edge;
  class Mesh;
  class MeshEditor;
  template<class T> class MeshFunction;

  /// This class implements local mesh refinement for different mesh types.

  class LocalMeshRefinement
  {
  public:

    /// Refine simplicial mesh locally by edge bisection
    static void refineMeshByEdgeBisection(Mesh& refined_mesh,
                                          const Mesh& mesh,
                                          const MeshFunction<bool>& cell_marker,
                                          bool refine_boundary = true);

    /// Iteratively refine mesh locally by the longest edge bisection
    static void refineIterativelyByEdgeBisection(Mesh& refined_mesh,
                                                 const Mesh& mesh,
                                                 const MeshFunction<bool>& cell_marker);

    /// Recursively refine mesh locally by the longest edge bisection
    /// Fast Rivara algorithm implementation with propagation of MeshFunctions and
    /// arrays for boundary indicators
    static void refineRecursivelyByEdgeBisection(Mesh& refined_mesh,
                                                 const Mesh& mesh,
                                                 const MeshFunction<bool>& cell_marker);

  private:

    /// Bisect edge of simplex cell
    static void bisect_simplex_edge(const Cell& cell, const Edge& edge,
                                    uint new_vertex, MeshEditor& editor,
                                    uint& current_cell);

    /// Iteration of iterative algorithm
    static bool iteration_of_refinement(Mesh& mesh,
                                      const MeshFunction<bool>& cell_marker,
                                      MeshFunction<bool>& new_cell_marker,
                                      MeshFunction<uint>& bisected_edges);

    /// Transform mesh data
    static void transform_data(Mesh& newmesh, const Mesh& oldmesh,
                               const MeshFunction<uint>& cell_map,
		                           const std::vector<int>& facet_map);

  };

}

#endif
