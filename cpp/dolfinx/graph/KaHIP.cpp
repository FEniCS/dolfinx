// Copyright (C) 2019 Igor A. Baratta
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "KaHIP.h"
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <map>

#ifdef HAS_KAHIP
#include <parhip_interface.h>
#endif

using namespace dolfinx;

#ifdef HAS_KAHIP

graph::AdjacencyList<std::int32_t> dolfinx::graph::KaHIP::partition(
    MPI_Comm mpi_comm, int nparts,
    const graph::AdjacencyList<unsigned long long>& adj_graph, bool ghosting)
{
  common::Timer timer("Compute graph partition (KaHIP)");

  const std::int32_t num_processes = dolfinx::MPI::size(mpi_comm);
  const std::int32_t process_number = dolfinx::MPI::rank(mpi_comm);

  // Graph does not have vertex or adjacency weights, so we use null
  // pointers as arguments.
  unsigned long long* vwgt{nullptr};
  unsigned long long* adjcwgt{nullptr};

  // TODO: Allow the user to set the parameters
  int mode = 4; // Fast Mode
  int seed = 0;

  // The amount of imbalance that is allowed. (3%)
  double imbalance = 0.03;

  // Suppress output from the partitioning library.
  bool suppress_output = true;

  // Call KaHIP to partition graph
  common::Timer timer1("KaHIP: call ParHIPPartitionKWay");

  std::vector<unsigned long long> node_distribution(num_processes);
  const unsigned long long num_local_cells = adj_graph.num_nodes();
  MPI_Allgather(&num_local_cells, 1, MPI::mpi_type<unsigned long long>(),
                node_distribution.data(), 1,
                MPI::mpi_type<unsigned long long>(), mpi_comm);

  node_distribution.insert(node_distribution.begin(), 0);
  for (std::size_t i = 1; i != node_distribution.size(); ++i)
    node_distribution[i] += node_distribution[i - 1];

  std::vector<unsigned long long> part(num_local_cells);
  std::vector<unsigned long long> adj_graph_offsets(
      adj_graph.offsets().data(),
      adj_graph.offsets().data() + adj_graph.offsets().size());
  int edgecut = 0;

  ParHIPPartitionKWay(const_cast<unsigned long long*>(node_distribution.data()),
                      const_cast<unsigned long long*>(adj_graph_offsets.data()),
                      const_cast<unsigned long long*>(adj_graph.array().data()),
                      vwgt, adjcwgt, &nparts, &imbalance, suppress_output, seed,
                      mode, &edgecut, part.data(), &mpi_comm);
  timer1.stop();

  const unsigned long long elm_begin = node_distribution[process_number];
  const unsigned long long elm_end = node_distribution[process_number + 1];
  const std::int32_t ncells = elm_end - elm_begin;

  // Create a map of local nodes to their additional destination processes,
  // due to ghosting. If no ghosting, this will remain empty.
  std::map<std::int32_t, std::set<std::int32_t>> local_node_to_dests;
  if (ghosting)
  {
    common::Timer timer2("Compute graph halo data (KaHIP)");

    // Work out halo cells for current division of dual graph
    std::map<unsigned long long, std::set<std::int32_t>> halo_cell_to_remotes;
    // local indexing "i"
    for (int i = 0; i < ncells; i++)
    {
      for (int j = 0; j < adj_graph.num_links(i); ++j)
      {
        const unsigned long long other_cell = adj_graph.links(i)[j];
        if (other_cell < elm_begin || other_cell >= elm_end)
        {
          const int remote
              = std::upper_bound(node_distribution.begin(),
                                 node_distribution.end(), other_cell)
                - node_distribution.begin() - 1;

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
      for (auto proc : hcell.second)
      {
        assert(proc < num_processes);

        // global cell number
        send_cell_partition[proc].push_back(hcell.first + elm_begin);

        // partitioning
        send_cell_partition[proc].push_back(part[hcell.first]);
      }
    }

    // Actual halo exchange
    const Eigen::Array<std::int64_t, Eigen::Dynamic, 1> recv_cell_partition
        = dolfinx::MPI::all_to_all(
              mpi_comm, graph::AdjacencyList<std::int64_t>(send_cell_partition))
              .array();

    // Construct a map from all currently foreign cells to their new
    // partition number
    std::map<std::int64_t, std::int32_t> cell_ownership;
    for (int p = 0; p < recv_cell_partition.rows(); p += 2)
      cell_ownership[recv_cell_partition[p]] = recv_cell_partition[p + 1];

    const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& xadj
        = adj_graph.offsets();
    const Eigen::Array<unsigned long long, Eigen::Dynamic, 1>& adjncy
        = adj_graph.array();

    // Generate map for where new boundary cells need to be sent
    for (std::int32_t i = 0; i < ncells; i++)
    {
      const std::int32_t proc_this = part[i];
      for (std::int32_t j = xadj[i]; j < xadj[i + 1]; ++j)
      {
        const unsigned long long other_cell = adjncy[j];
        std::int32_t proc_other;

        if (other_cell < elm_begin || other_cell >= elm_end)
        { // remote cell - should be in map
          const auto find_other_proc = cell_ownership.find(other_cell);
          assert(find_other_proc != cell_ownership.end());
          proc_other = find_other_proc->second;
        }
        else
          proc_other = part[other_cell - elm_begin];

        if (proc_this != proc_other)
          local_node_to_dests[i].insert(proc_other);
      }
    }
    timer2.stop();
  }

  // Convert to offset format for AdjacencyList
  std::vector<std::int32_t> dests;
  std::vector<std::int32_t> offsets = {0};
  for (std::int32_t i = 0; i < ncells; ++i)
  {
    dests.push_back(part[i]);
    const auto it = local_node_to_dests.find(i);
    if (it != local_node_to_dests.end())
      dests.insert(dests.end(), it->second.begin(), it->second.end());
    offsets.push_back(dests.size());
  }

  return graph::AdjacencyList<std::int32_t>(dests, offsets);
}

#endif
