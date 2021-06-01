// Copyright (C) 2019 Igor A. Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "kahip.h"
#include <algorithm>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <map>
#include <set>
#include <vector>

#ifdef HAS_KAHIP
#include <parhip_interface.h>
#endif

using namespace dolfinx;

#ifdef HAS_KAHIP

//----------------------------------------------------------------------------
std::function<graph::AdjacencyList<std::int32_t>(
    MPI_Comm, int, const graph::AdjacencyList<std::int64_t>&, std::int32_t,
    bool)>
graph::kahip::partitioner(int mode, int seed, double imbalance,
                          bool suppress_output)
{
  return [mode, seed, imbalance,
          suppress_output](MPI_Comm mpi_comm, int nparts,
                           const graph::AdjacencyList<std::int64_t>& graph,
                           std::int32_t, bool ghosting) {
    common::Timer timer("Compute graph partition (KaHIP)");

    const auto& local_graph = graph.as_type<unsigned long long>();
    const int num_processes = dolfinx::MPI::size(mpi_comm);
    const int process_number = dolfinx::MPI::rank(mpi_comm);

    // Graph does not have vertex or adjacency weights, so we use null
    // pointers as arguments
    unsigned long long *vwgt(nullptr), *adjcwgt(nullptr);

    // Call KaHIP to partition graph
    common::Timer timer1("KaHIP: call ParHIPPartitionKWay");

    // Compute distribution across all ranks
    std::vector<unsigned long long> node_dist(num_processes + 1, 0);
    const unsigned long long num_local_nodes = local_graph.num_nodes();
    MPI_Allgather(&num_local_nodes, 1, MPI_UNSIGNED_LONG_LONG,
                  node_dist.data() + 1, 1, MPI_UNSIGNED_LONG_LONG, mpi_comm);
    std::partial_sum(node_dist.begin(), node_dist.end(), node_dist.begin());

    // Partition graph
    std::vector<unsigned long long> part(num_local_nodes);
    std::vector<unsigned long long> adj_graph_offsets(
        local_graph.offsets().begin(), local_graph.offsets().end());
    int edgecut = 0;
    double _imbalance = imbalance;
    ParHIPPartitionKWay(
        const_cast<unsigned long long*>(node_dist.data()),
        const_cast<unsigned long long*>(adj_graph_offsets.data()),
        const_cast<unsigned long long*>(local_graph.array().data()), vwgt,
        adjcwgt, &nparts, &_imbalance, suppress_output, seed, mode, &edgecut,
        part.data(), &mpi_comm);
    timer1.stop();

    const unsigned long long elm_begin = node_dist[process_number];
    const unsigned long long elm_end = node_dist[process_number + 1];
    const std::int32_t ncells = elm_end - elm_begin;

    // Create a map of local nodes to their additional destination
    // processes, due to ghosting. If no ghosting, this will remain empty.
    std::map<std::int32_t, std::set<std::int32_t>> local_node_to_dests;
    if (ghosting)
    {
      common::Timer timer2("Compute graph halo data (KaHIP)");

      // Work out halo cells for current division of dual graph
      std::map<unsigned long long, std::set<std::int32_t>> halo_cell_to_remotes;
      // local indexing "i"
      for (int i = 0; i < ncells; i++)
      {
        const auto edges = local_graph.links(i);
        for (std::size_t j = 0; j < edges.size(); ++j)
        {
          const unsigned long long other_cell = edges[j];
          if (other_cell < elm_begin or other_cell >= elm_end)
          {
            auto it = std::upper_bound(node_dist.begin(), node_dist.end(),
                                       other_cell);
            const int remote = std::distance(node_dist.begin(), it) - 1;
            assert(remote < num_processes);
            if (halo_cell_to_remotes.find(i) == halo_cell_to_remotes.end())
              halo_cell_to_remotes[i] = std::set<std::int32_t>();
            halo_cell_to_remotes[i].insert(remote);
          }
        }
      }

      // Do halo exchange of cell partition data
      std::vector<std::vector<std::int64_t>> send_cell_partition(num_processes);
      for (const auto& hcell : halo_cell_to_remotes)
      {
        for (std::int64_t proc : hcell.second)
        {
          assert(proc < num_processes);

          // global cell number
          send_cell_partition[proc].push_back(hcell.first + elm_begin);

          // partitioning
          send_cell_partition[proc].push_back(part[hcell.first]);
        }
      }

      // Actual halo exchange
      const std::vector<std::int64_t> recv_cell_partition
          = dolfinx::MPI::all_to_all(
                mpi_comm,
                graph::AdjacencyList<std::int64_t>(send_cell_partition))
                .array();

      // Construct a map from all currently foreign cells to their new
      // partition number
      std::map<std::int64_t, std::int32_t> cell_ownership;
      for (std::size_t p = 0; p < recv_cell_partition.size(); p += 2)
        cell_ownership[recv_cell_partition[p]] = recv_cell_partition[p + 1];

      const std::vector<std::int32_t>& xadj = local_graph.offsets();
      const std::vector<unsigned long long>& adjncy = local_graph.array();

      // Generate map for where new boundary cells need to be sent
      for (std::int32_t i = 0; i < ncells; i++)
      {
        const std::int32_t proc_this = part[i];
        for (std::int32_t j = xadj[i]; j < xadj[i + 1]; ++j)
        {
          const unsigned long long other_cell = adjncy[j];
          if (other_cell < elm_begin or other_cell >= elm_end)
          { // remote cell - should be in map
            const auto it = cell_ownership.find(other_cell);
            assert(it != cell_ownership.end());
            if (it->second != proc_this)
              local_node_to_dests[i].insert(it->second);
          }
          else
          {
            if ((int)part[other_cell - elm_begin] != proc_this)
              local_node_to_dests[i].insert(part[other_cell - elm_begin]);
          }
        }
      }
      timer2.stop();
    }

    // Convert to offset format for AdjacencyList
    std::vector<std::int32_t> dests;
    dests.reserve(ncells);
    std::vector<std::int32_t> offsets(1, 0);
    for (std::int32_t i = 0; i < ncells; ++i)
    {
      dests.push_back(part[i]);
      if (auto it = local_node_to_dests.find(i);
          it != local_node_to_dests.end())
      {
        dests.insert(dests.end(), it->second.begin(), it->second.end());
      }
      offsets.push_back(dests.size());
    }

    return graph::AdjacencyList<std::int32_t>(std::move(dests),
                                              std::move(offsets));
  };
}
//----------------------------------------------------------------------------
#endif
