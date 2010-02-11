// Copyright (C) 2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-08-17
// Last changed:

#ifndef __GRAPH_BUILDER_H
#define __GRAPH_BUILDER_H

#include <set>
#include <vector>
#include <dolfin/common/types.h>

namespace dolfin
{

  // Forward declarations
  class LocalMeshData;
  class Mesh;

  /// This class builds a Graph corresponding for various objects (Mesh, matrix
  /// sparsity pattern, etc)

  class GraphBuilder
  {

  public:

    /// Build Graph of a mesh
    static void build(LocalMeshData& mesh_data);

  private:

    static void compute_connectivity(const std::vector<std::vector<uint> >& cell_vertices,
                                     uint num_cell_facets, uint num_facet_vertices,
                                     uint offset,
                                     std::vector<std::set<uint> >& graph);

    static uint compute_connectivity(const std::vector<std::vector<uint> >& cell_vertices,
                                     const std::vector<std::vector<uint> >& candidate_ghost_vertices,
                                     const std::vector<uint>& candidate_ghost_global_indices,
                                     uint num_cell_facets, uint num_facet_vertices,
                                     std::vector<std::set<uint> >& ghost_graph_edges,
                                     std::set<uint>& ghost_cells);

    static void compute_scotch_data(const std::vector<std::set<uint> >& graph_edges,
                                    const std::set<uint>& ghost_cells,
                                    uint num_global_vertices);

  };

}

#endif
