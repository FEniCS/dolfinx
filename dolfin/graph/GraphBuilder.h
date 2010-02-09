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
#include "Graph.h"

namespace dolfin
{

  // Forward declarations
  class Graph;
  class LocalMeshData;
  class Mesh;

  /// This class builds a Graph corresponding for various objects (Mesh, matrix
  /// sparsity pattern, etc)

  class GraphBuilder
  {

  public:

    /// Build Graph of a mesh
    static void build(Graph& graph, LocalMeshData& mesh_data, 
                      Graph::Representation rep = Graph::dual);

    /// Build Graph of a mesh
    static void build(Graph& graph, const Mesh& mesh, 
                      Graph::Representation rep = Graph::dual);

  private:

    static void create_nodal(Graph& graph, const Mesh& mesh);

    static void create_dual(Graph& graph, const Mesh& mesh);

    static void compute_connectivity(const std::vector<std::vector<uint> >& cell_vertices,
                                     const std::vector<uint>& global_cell_indices,
                                     uint num_cell_facets, uint num_facet_vertices,
                                     std::vector<std::set<uint> >& graph_edges);

    static void compute_connectivity(const std::vector<std::vector<uint> >& cell_vertices,
                                     const std::vector<std::vector<uint> >& ghost_cell_vertices,
                                     const std::vector<uint>& ghost_global_cell_indices,
                                     uint num_cell_facets, uint num_facet_vertices,
                                     std::vector<std::set<uint> >& graph_edges);
  };

}

#endif
