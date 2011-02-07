// Copyright (C) 2011 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2011-02-07
// Last changed: 2011-02-07

#ifndef __REGULAR_CUT_REFINEMENT_H
#define __REGULAR_CUT_REFINEMENT_H

#include <vector>

namespace dolfin
{

  class Mesh;
  template<class T> class MeshFunction;

  /// This class implements local mesh refinement by a regular cut of
  /// each cell marked for refinement in combination with propagation
  /// of cut edges to neighboring cells. Losely based on the paper
  /// "Tetrahedral grid refinement" by Jurgen Bey (1992).

  class RegularCutRefinement
  {
  public:

    /// Refine mesh based on cell markers
    static void refine(Mesh& refined_mesh,
                       const Mesh& mesh,
                       const MeshFunction<bool>& cell_markers);

  private:

    // Compute refinement markers based on initial markers
    static void compute_markers(const Mesh& mesh,
                                const MeshFunction<bool>& cell_markers);

    // Count the number of marked entries
    static uint count_markers(const std::vector<bool>& markers);

    // Mark edge for refinement and add cell to list of marked cells
    // if edge has not been marked before
    static void mark(std::vector<bool>& edge_markers,
                     uint cell_index,
                     uint local_edge_index,
                     std::vector<uint>& marked_cells);
  };

}

#endif

