// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-02-19
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

  /// This class builds a Graph corresponding to various objects

  class GraphBuilder
  {

  public:

    /// Build distributed dual graph for mesh
    static void compute_dual_graph(const LocalMeshData& mesh_data,
                                   std::vector<std::set<uint> >& local_graph,
                                   std::set<uint>& ghost_vertices);

  private:

    static void compute_connectivity(const std::vector<std::vector<uint> >& cell_vertices,
                                     uint num_facet_vertices, uint offset,
                                     std::vector<std::set<uint> >& graph);

    static uint compute_ghost_connectivity(const std::vector<std::vector<uint> >& cell_vertices,
                                     const std::vector<uint>& local_boundary_cells,
                                     const std::vector<std::vector<uint> >& candidate_ghost_vertices,
                                     const std::vector<uint>& candidate_ghost_global_indices,
                                     uint num_facet_vertices,
                                     std::vector<std::set<uint> >& ghost_graph_edges,
                                     std::set<uint>& ghost_cells);

  };

}

#endif
