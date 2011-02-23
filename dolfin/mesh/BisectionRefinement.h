// Copyright (C) 2006 Johan Hoffman.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2010-2011.
// Modified by Anders Logg, 2010-2011.
//
// First added:  2006-11-01
// Last changed: 2011-02-22

#ifndef __BISECTION_REFINEMENT_H
#define __BISECTION_REFINEMENT_H

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

  /// This class implements local mesh refinement by edge bisection.

  class BisectionRefinement
  {
  public:

    /// Recursively refine mesh locally by longest edge bisection
    /// (Rivara). Fast Rivara algorithm implementation with
    /// propagation of MeshFunctions and arrays for boundary
    /// indicators.
    static void refine_by_recursive_bisection(Mesh& refined_mesh,
                                              const Mesh& mesh,
                                              const MeshFunction<bool>& cell_marker);

  private:

    /// Transform mesh data
    static void transform_data(Mesh& newmesh, const Mesh& oldmesh,
                               const MeshFunction<uint>& cell_map,
                               const std::vector<int>& facet_map);

  };

}

#endif
