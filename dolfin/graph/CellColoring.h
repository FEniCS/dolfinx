// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-11-15
// Last changed: 2010-11-16

#ifndef __DOLFIN_ZOLTAN_CELL_COLORING_H
#define __DOLFIN_ZOLTAN_CELL_COLORING_H

#ifdef HAS_TRILINOS

#include <string>
#include <vector>
#include <boost/unordered_set.hpp>
#include <zoltan_cpp.h>
#include <dolfin/common/types.h>
#include <dolfin/mesh/Cell.h>
#include "graph_types.h"

namespace dolfin
{

  class Mesh;

  /// This class computes cell colorings for a local mesh. It supports vertex,
  /// facet and edge-based colorings.  Zoltan (part of Trilinos) is used to
  /// the colorings.

  class CellColoring
  {

  public:

    /// Constructor
    CellColoring(const Mesh& mesh, std::string type="vertex");

    /// Compute cell colors
    CellFunction<uint> compute_local_cell_coloring();

  private:

    // Build graph that is to be colored
    template<class T> void build_graph(const Mesh& mesh, Graph& graph);

    // Mesh
    const Mesh& mesh;

    // Graph (cell neighbours)
    Graph graph;

  };

}

#endif
#endif
