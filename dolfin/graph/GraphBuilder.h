// Copyright (C) 2010-2013 Garth N. Wells
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
// Last changed: 2013-01-16

#ifndef __GRAPH_BUILDER_H
#define __GRAPH_BUILDER_H

#include <set>
#include <boost/unordered_map.hpp>
#include <vector>
#include <boost/multi_array.hpp>
#include <dolfin/common/MPI.h>
#include "Graph.h"

namespace dolfin
{

  // Forward declarations
  class CellType;
  class GenericDofMap;
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
                             const std::vector<std::size_t>& coloring_type);

    /// Build local graph (specialized version)
    static Graph local_graph(const Mesh& mesh, std::size_t dim0,
                             std::size_t dim1);

    /// Build distributed dual graph (cell-cell connections) from
    /// minimal mesh data
    static void
      compute_dual_graph(const MPI_Comm mpi_comm,
                         const boost::multi_array<std::size_t, 2>& cell_vertices,
                         const CellType& cell_type,
                         const std::vector<std::size_t>& global_cell_indices,
                         const std::int64_t num_global_vertices,
                         std::vector<std::set<std::size_t>>& local_graph,
                         std::set<std::size_t>& ghost_vertices);

  private:

    friend class MeshPartitioning;

    typedef boost::unordered_map<std::vector<std::size_t>, std::size_t>
      FacetCellMap;

    static void
      compute_local_dual_graph(const MPI_Comm mpi_comm,
                               const boost::multi_array<std::size_t, 2>& cell_vertices,
                               const CellType& cell_type,
                               const std::vector<std::size_t>& global_cell_indices,
                               std::vector<std::set<std::size_t>>& local_graph,
                               FacetCellMap& facet_cell_map);

    // Build nonlocal part of dual graph for mesh.
    // GraphBuilder::compute_local_dual_graph should be called first.
    static void
      compute_nonlocal_dual_graph(const MPI_Comm mpi_comm,
                                  const boost::multi_array<std::size_t, 2>& cell_vertices,
                                  const CellType& cell_type,
                                  const std::vector<std::size_t>& global_cell_indices,
                                  const std::int64_t num_global_vertices,
                                  std::vector<std::set<std::size_t>>& local_graph,
                                  FacetCellMap& facet_cell_map,
                                  std::set<std::size_t>& ghost_vertices);

  };

}

#endif
