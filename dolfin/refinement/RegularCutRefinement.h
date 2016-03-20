// Copyright (C) 2011 Anders Logg
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
// First added:  2011-02-07
// Last changed: 2011-02-09

#ifndef __REGULAR_CUT_REFINEMENT_H
#define __REGULAR_CUT_REFINEMENT_H

#include <vector>

namespace dolfin
{

  class Cell;
  class Mesh;
  template<typename T> class MeshFunction;
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
    enum class marker_type : int { no_refinement=-1, regular_refinement=-2,
        backtrack_bisection=-3, backtrack_bisection_refine=-4 };

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
    static std::size_t count_markers(const std::vector<bool>& markers);

    // Extract index of first marked entry
    static std::size_t extract_edge(const std::vector<bool>& markers);

    // Check whether suggested refinement will produce too thin cells
    static bool too_thin(const Cell& cell,
                         const std::vector<bool>& edge_markers);

    // Find local indices for common edge relative to cell and twin
    static std::pair<std::size_t, std::size_t> find_common_edges(const Cell& cell,
                                                   const Mesh& mesh,
                                                   std::size_t bisection_twin);

    // Find local indices for bisection edge relative to cell and twin
    static std::pair<std::size_t, std::size_t> find_bisection_edges(const Cell& cell,
                                                      const Mesh& mesh,
                                                      std::size_t bisection_twin);

    // Find local indices for bisection vertex relative to cell and twin
    static std::pair<std::size_t, std::size_t> find_bisection_vertices(const Cell& cell,
                                                         const Mesh& mesh,
                                                         std::size_t bisection_twin,
                                                         const std::pair<std::size_t, std::size_t>& bisection_edges);

  };

}

#endif
