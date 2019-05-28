// Copyright (C) 2008-2016 Igor A. Baratta
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "KaHIP.h"
#include "CSRGraph.h"
#include "Graph.h"
#include "GraphBuilder.h"
#include <dolfin/common/MPI.h>
#include <dolfin/common/Timer.h>
#include <dolfin/mesh/CellType.h>

#ifdef HAS_KAHIP
#include <parhip_interface.h>
#endif

using namespace dolfin;

#ifdef HAS_KAHIP

void dolfin::graph::KaHIP::partition(
    MPI_Comm mpi_comm, const CSRGraph<unsigned long long>& csr_graph)
{
  std::map<std::int64_t, std::vector<int>> ghost_procs;
  common::Timer timer("Compute graph partition (KaHIP)");

  // Number of partitions (one for each process)
  // FIXME: Allow partition on a subset of processes
  int nparts = dolfin::MPI::size(mpi_comm);

  // Graph does not have vertex or adjacency weights,
  // so we use null pointers as arguments.
  idxtype* vwgt{nullptr};
  idxtype* adjcwgt{nullptr};

  // The amount of imbalance that is allowed. (3%)
  double imbalance = 0.03;

  // Suppress output from the partitioning library.
  bool suppress_output = true;

  // FIXME: Allow the user to set
  int mode = ULTRAFASTMESH;
  int seed = 0;

  // Call KaHIP to partition graph
  common::Timer timer1("KaHIP: call ParHIPPartitionKWay");
  const std::int32_t num_local_cells = csr_graph.size();
  std::vector<idxtype> part(num_local_cells);
  int edgecut = 0;
  assert(!part.empty());

  // void ParHIPPartitionKWay(idxtype *vtxdist, idxtype *xadj, idxtype *adjncy,
  //                          idxtype *vwgt, idxtype *adjwgt,
  //                          int *nparts, double* imbalance, bool
  //                          suppress_output, int seed, int mode, int *edgecut,
  //                          idxtype *part, MPI_Comm *comm);
  //
  ParHIPPartitionKWay(
      const_cast<idxtype*>(csr_graph.node_distribution().data()),
      const_cast<idxtype*>(csr_graph.nodes().data()),
      const_cast<idxtype*>(csr_graph.edges().data()), vwgt, adjcwgt, &nparts,
      &imbalance, suppress_output, seed, mode, &edgecut, part.data(),
      &mpi_comm);
  timer1.stop();

  common::Timer timer2("Compute graph halo data (KaHIP)");
  //
  // Work out halo cells for current division of dual graph
  // const auto& elmdist = csr_graph.node_distribution();
  // const auto& xadj = csr_graph.nodes();
  // const auto& adjncy = csr_graph.edges();
  // const std::int32_t num_processes = dolfin::MPI::size(mpi_comm);
  // const std::int32_t process_number = dolfin::MPI::rank(mpi_comm);
  // const idx_t elm_begin = elmdist[process_number];
  // const idx_t elm_end = elmdist[process_number + 1];
  // const std::int32_t ncells = elm_end - elm_begin;
  //
  // std::map<idx_t, std::set<std::int32_t>> halo_cell_to_remotes;
  // // local indexing "i"
  // for (int i = 0; i < ncells; i++)
  // {
  //   for (auto other_cell :
  //        csr_graph[i]) // idx_t j = xadj[i]; j != xadj[i + 1]; ++j)
  //   {
  //     //      const idx_t other_cell = adjncy[j];
  //     if (other_cell < elm_begin || other_cell >= elm_end)
  //     {
  //       const int remote
  //           = std::upper_bound(elmdist.begin(), elmdist.end(), other_cell)
  //             - elmdist.begin() - 1;
  //
  //       assert(remote < num_processes);
  //       if (halo_cell_to_remotes.find(i) == halo_cell_to_remotes.end())
  //         halo_cell_to_remotes[i] = std::set<std::int32_t>();
  //       halo_cell_to_remotes[i].insert(remote);
  //     }
  //   }
  // }
  //
  // // Do halo exchange of cell partition data
  // std::vector<std::vector<std::int64_t>> send_cell_partition(num_processes);
  // std::vector<std::int64_t> recv_cell_partition;
  // for (const auto& hcell : halo_cell_to_remotes)
  // {
  //   for (auto proc : hcell.second)
  //   {
  //     assert(proc < num_processes);
  //
  //     // global cell number
  //     send_cell_partition[proc].push_back(hcell.first + elm_begin);
  //
  //     // partitioning
  //     send_cell_partition[proc].push_back(part[hcell.first]);
  //   }
  // }
  //
  // // Actual halo exchange
  // dolfin::MPI::all_to_all(mpi_comm, send_cell_partition,
  // recv_cell_partition);
  //
  // // Construct a map from all currently foreign cells to their new
  // // partition number
  // std::map<std::int64_t, std::int32_t> cell_ownership;
  // for (auto p = recv_cell_partition.begin(); p != recv_cell_partition.end();
  //      p += 2)
  // {
  //   cell_ownership[*p] = *(p + 1);
  // }
  //
  // // Generate mapping for where new boundary cells need to be sent
  // for (std::int32_t i = 0; i < ncells; i++)
  // {
  //   const std::size_t proc_this = part[i];
  //   for (idx_t j = xadj[i]; j < xadj[i + 1]; ++j)
  //   {
  //     const idx_t other_cell = adjncy[j];
  //     std::size_t proc_other;
  //
  //     if (other_cell < elm_begin || other_cell >= elm_end)
  //     { // remote cell - should be in map
  //       const auto find_other_proc = cell_ownership.find(other_cell);
  //       assert(find_other_proc != cell_ownership.end());
  //       proc_other = find_other_proc->second;
  //     }
  //     else
  //       proc_other = part[other_cell - elm_begin];
  //
  //     if (proc_this != proc_other)
  //     {
  //       auto map_it = ghost_procs.find(i);
  //       if (map_it == ghost_procs.end())
  //       {
  //         std::vector<std::int32_t> sharing_processes;
  //         sharing_processes.push_back(proc_this);
  //         sharing_processes.push_back(proc_other);
  //         ghost_procs.insert({i, sharing_processes});
  //       }
  //       else
  //       {
  //         // Add to vector if not already there
  //         auto it = std::find(map_it->second.begin(), map_it->second.end(),
  //                             proc_other);
  //         if (it == map_it->second.end())
  //           map_it->second.push_back(proc_other);
  //       }
  //     }
  //   }
  // }
  //
  timer2.stop();
  //
  // return std::make_pair(std::vector<int>(part.begin(), part.end()),
  //                       std::move(ghost_procs));
}

#endif
