// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2010.
//
// First added:  2010-11-15
// Last changed: 2010-11-27

#ifndef __MESH_COLORING_H
#define __MESH_COLORING_H

#include <string>
#include <dolfin/common/types.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/graph/Graph.h>

namespace dolfin
{

  class Mesh;

  /// This class computes colorings for a local mesh. It supports
  /// vertex, edge, and facet-based colorings.

  class MeshColoring
  {
  public:

    /// Color the cells of a mesh for given coloring type, which can
    /// be one of "vertex", "edge" or "facet".
    static const MeshFunction<uint>& color_cells(Mesh& mesh,
                                                 std::string coloring_type);

    /// Color the cells of a mesh for given coloring type specified by
    /// topological dimension, which can be one of 0, 1 or D - 1.
    static const MeshFunction<uint>& color(Mesh& mesh, uint colored_entity_dim,
                                           uint dim);

    /// Compute cell colors for given coloring type specified by
    /// topological dimension, which can be one of 0, 1 or D - 1.
    static void compute_colors(MeshFunction<uint>& colors,
                               uint colored_entity, uint dim);

    /// Convert coloring type to topological dimension
    static uint type_to_dim(std::string coloring_type, const Mesh& mesh);

  };

}

#endif
