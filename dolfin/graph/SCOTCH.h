// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-02-10
// Last changed:

#ifndef __SCOTCH_PARTITIONER_H
#define __SCOTCH_PARTITIONER_H

#include <set>
#include <vector>
#include <dolfin/common/types.h>

namespace dolfin
{
  // Forward declarations
  class LocalMeshData;

  /// This class proivdes an interface to SCOTCH-PT (parallel version)

  class SCOTCH
  {
  public:

    // Compute cell partition
    static void compute_partition(std::vector<uint>& cell_partition,
                                  const LocalMeshData& mesh_data);

  private:

    // Compute distribted dual graph for mesh
    static void compute_dual_graph(const LocalMeshData& mesh_data,
                                   std::vector<std::set<uint> >& local_graph,
                                   std::set<uint>& ghost_vertices);


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

    // Compute distribted dual graph for mesh
    static void partition(const std::vector<std::set<uint> >& local_graph,
                          const std::set<uint>& ghost_vertices,
                          const std::vector<uint>& global_cell_indices,
                          uint num_global_vertices,
                          std::vector<uint>& cell_partition);

  };

}

#endif
