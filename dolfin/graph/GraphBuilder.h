// Copyright (C) 2010-2011 Garth N. Wells
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
// First added:  2010-02-19
// Last changed: 2011-02-22

#ifndef __GRAPH_BUILDER_H
#define __GRAPH_BUILDER_H

#include <set>
#include <vector>
#include <boost/multi_array.hpp>
#include <dolfin/common/types.h>
#include "Graph.h"

namespace dolfin
{

  // Forward declarations
  class GenericDofMap;
  class LocalMeshData;
  class Mesh;

  /// This class builds a Graph corresponding to various objects

  class GraphBuilder
  {

  public:

    /// Build local graph from dofmap
    static Graph local_graph(const Mesh& mesh, const GenericDofMap& dofmap0,
                                               const GenericDofMap& dofmap1);

    /// Build local graph from mesh (general version)
    static Graph local_graph(const Mesh& mesh,
                             const std::vector<uint>& coloring_type);

    // Build local Boost graph (general version)
    static BoostBidirectionalGraph local_boost_graph(const Mesh& mesh,
                                        const std::vector<uint>& coloring_type);

    // Build local graph (specialized version)
    static Graph local_graph(const Mesh& mesh, uint dim0, uint dim1);

    // Build local Boost graph (specialized version)
    static BoostBidirectionalGraph local_boost_graph(const Mesh& mesh, uint dim0, uint dim1);

    /// Build distributed dual graph for mesh
    static void compute_dual_graph(const LocalMeshData& mesh_data,
                                   std::vector<std::set<std::size_t> >& local_graph,
                                   std::set<std::size_t>& ghost_vertices);

  private:

    static void compute_connectivity(const boost::multi_array<std::size_t, 2>& cell_vertices,
                                     uint num_facet_vertices, std::size_t offset,
                                     std::vector<std::set<std::size_t> >& graph);

    static std::size_t compute_ghost_connectivity(const boost::multi_array<std::size_t, 2>& cell_vertices,
                                     const std::vector<std::size_t>& local_boundary_cells,
                                     const std::vector<std::vector<std::size_t> >& candidate_ghost_vertices,
                                     const std::vector<std::size_t>& candidate_ghost_global_indices,
                                     uint num_facet_vertices,
                                     std::vector<std::set<std::size_t> >& ghost_graph_edges,
                                     std::set<std::size_t>& ghost_cells);

  };

}

#endif
