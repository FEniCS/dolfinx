// Copyright (C) 2011 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2011-02-07
// Last changed: 2011-02-09

#ifndef __REGULAR_CUT_REFINEMENT_H
#define __REGULAR_CUT_REFINEMENT_H

#include <vector>

namespace dolfin
{

  class Mesh;
  template<class T> class MeshFunction;
  class IndexSet;

  /// This class implements local mesh refinement by a regular cut of
  /// each cell marked for refinement in combination with propagation
  /// of cut edges to neighboring cells.

  class RegularCutRefinement
  {
  public:

    /// Refine mesh based on cell markers
    static void refine(Mesh& refined_mesh,
                       const Mesh& mesh,
                       const MeshFunction<bool>& cell_markers);

  private:

    // Refinement markers
    enum { no_refinement=-1, regular_refinement=-2, backtrack_bisection=-3, backtrack_bisection_refine=-4 };

    // Compute refinement markers based on initial markers
    static void compute_markers(std::vector<int>& refinement_markers,
                                IndexSet& marked_edges,
                                const Mesh& mesh,
                                const MeshFunction<bool>& cell_markers);

    // Refine mesh based on computed markers
    static void refine_marked(Mesh& refined_mesh,
                              const Mesh& mesh,
                              const std::vector<int>& refinement_markers,
                              const IndexSet& marked_edges);

    // Count the number of marked entries
    static uint count_markers(const std::vector<bool>& markers);

    // Extract index of first marked entry
    static uint extract_edge(const std::vector<bool>& markers);

    // Check whether suggested refinement will produce too thin cells
    static bool too_thin(const Cell& cell,
                         const std::vector<bool>& edge_markers);

    // Find local indices for common edge relative to cell and twin
    static std::pair<uint, uint> find_common_edges(const Cell& cell,
                                                   const Mesh& mesh,
                                                   uint bisection_twin);

    // Find local indices for bisection edge relative to cell and twin
    static std::pair<uint, uint> find_bisection_edges(const Cell& cell,
                                                      const Mesh& mesh,
                                                      uint bisection_twin);

    // Find local indices for bisection vertex relative to cell and twin
    static std::pair<uint, uint> find_bisection_vertices(const Cell& cell,
                                                         const Mesh& mesh,
                                                         uint bisection_twin,
                                                         const std::pair<uint, uint>& bisection_edges);

  };

}

#endif
