// Copyright (C) 2008-2016 Niclas Jansson, Ola Skavhaug, Anders Logg,
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "parmetis.h"
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <map>
#include <set>
#include <vector>

#ifdef HAS_PARMETIS
#include <parmetis.h>
#endif

using namespace dolfinx;

#ifdef HAS_PARMETIS
namespace
{
//-----------------------------------------------------------------------------
template <typename T>
std::vector<int> adaptive_repartition(MPI_Comm mpi_comm,
                                      const graph::AdjacencyList<T>& adj_graph,
                                      double weight)
{
  common::Timer timer(
      "Compute graph partition (ParMETIS Adaptive Repartition)");

  // Options for ParMETIS
  idx_t options[4];
  options[0] = 1;
  options[1] = 0;
  options[2] = 15;
  options[3] = PARMETIS_PSR_UNCOUPLED;
  // For repartition, PARMETIS_PSR_COUPLED seems to suppress all
  // migration if already balanced.  Try PARMETIS_PSR_UNCOUPLED for
  // better edge cut.

  common::Timer timer1("ParMETIS: call ParMETIS_V3_AdaptiveRepart");
  real_t _itr = weight;
  std::vector<idx_t> part(adj_graph.num_nodes());
  std::vector<idx_t> vsize(part.size(), 1);
  assert(!part.empty());

  // Number of partitions (one for each process)
  idx_t nparts = dolfinx::MPI::size(mpi_comm);

  // Remaining ParMETIS parameters
  idx_t ncon = 1;
  idx_t* elmwgt = nullptr;
  idx_t wgtflag = 0;
  idx_t edgecut = 0;
  idx_t numflag = 0;
  std::vector<real_t> tpwgts(ncon * nparts, 1.0 / static_cast<real_t>(nparts));
  std::vector<real_t> ubvec(ncon, 1.05);

  // Call ParMETIS to repartition graph
  int err = ParMETIS_V3_AdaptiveRepart(
      adj_graph.node_distribution().data(), adj_graph.nodes().data(),
      adj_graph.edges().data(), elmwgt, nullptr, vsize.data(), &wgtflag,
      &numflag, &ncon, &nparts, tpwgts.data(), ubvec.data(), &_itr, options,
      &edgecut, part.data(), &mpi_comm);
  assert(err == METIS_OK);
  timer1.stop();

  // Copy cell partition data and return
  return std::vector<int>(part.begin(), part.end());
}
//-----------------------------------------------------------------------------
template <typename T>
std::vector<int> refine(MPI_Comm mpi_comm,
                        const graph::AdjacencyList<T>& adj_graph)
{
  common::Timer timer("Compute graph partition (ParMETIS Refine)");

  // Get some MPI data
  const int process_number = dolfinx::MPI::rank(mpi_comm);

  // Options for ParMETIS
  idx_t options[4];
  options[0] = 1;
  options[1] = 0;
  options[2] = 15;
  // options[3] = PARMETIS_PSR_UNCOUPLED;

  // For repartition, PARMETIS_PSR_COUPLED seems to suppress all
  // migration if already balanced.  Try PARMETIS_PSR_UNCOUPLED for
  // better edge cut.

  // Partitioning array to be computed by ParMETIS. Prefill with
  // process_number.
  const std::int32_t num_local_cells = adj_graph.num_nodes();
  std::vector<idx_t> part(num_local_cells, process_number);
  assert(!part.empty());

  // Number of partitions (one for each process)
  idx_t nparts = dolfinx::MPI::size(mpi_comm);
  // Remaining ParMETIS parameters
  idx_t ncon = 1;
  idx_t* elmwgt = nullptr;
  idx_t wgtflag = 0;
  idx_t edgecut = 0;
  idx_t numflag = 0;
  std::vector<real_t> tpwgts(ncon * nparts, 1.0 / static_cast<real_t>(nparts));
  std::vector<real_t> ubvec(ncon, 1.05);

  // Call ParMETIS to partition graph
  common::Timer timer1("ParMETIS: call ParMETIS_V3_RefineKway");
  int err = ParMETIS_V3_RefineKway(
      adj_graph.node_distribution().data(), adj_graph.nodes().data(),
      adj_graph.edges().data(), elmwgt, nullptr, &wgtflag, &numflag, &ncon,
      &nparts, tpwgts.data(), ubvec.data(), options, &edgecut, part.data(),
      &mpi_comm);
  assert(err == METIS_OK);
  timer1.stop();

  // Copy cell partition data
  return std::vector<int>(part.begin(), part.end());
  //-----------------------------------------------------------------------------
}
} // namespace

//-----------------------------------------------------------------------------
graph::partition_fn graph::parmetis::partitioner(std::array<int, 3> options)
{
  return [options](MPI_Comm mpi_comm, idx_t nparts,
                   const graph::AdjacencyList<std::int64_t>& graph,
                   std::int32_t, bool ghosting) {
    LOG(INFO) << "Compute graph partition using ParMETIS";
    common::Timer timer("Compute graph partition (ParMETIS)");

    const auto& local_graph = graph.as_type<idx_t>();

    std::map<std::int64_t, std::vector<int>> ghost_procs;
    const int rank = dolfinx::MPI::rank(mpi_comm);
    const int size = dolfinx::MPI::size(mpi_comm);

    // Options for ParMETIS
    std::array<idx_t, 3> _options;
    std::copy(options.begin(), options.end(), _options.begin());

    // Strange weight arrays needed by ParMETIS
    idx_t ncon = 1;

    // Prepare remaining arguments for ParMETIS
    idx_t* elmwgt = nullptr;
    idx_t wgtflag = 0;
    idx_t edgecut = 0;
    idx_t numflag = 0;
    std::vector<real_t> tpwgts(ncon * nparts,
                               1.0 / static_cast<real_t>(nparts));
    std::vector<real_t> ubvec(ncon, 1.05);

    // Communicate number of nodes between all processors
    std::vector<idx_t> node_dist(size + 1, 0);
    const idx_t num_local_cells = local_graph.num_nodes();
    MPI_Allgather(&num_local_cells, 1, MPI::mpi_type<idx_t>(),
                  node_dist.data() + 1, 1, MPI::mpi_type<idx_t>(), mpi_comm);
    std::partial_sum(node_dist.begin(), node_dist.end(), node_dist.begin());

    // Note: ParMETIS is not const-correct, so we throw away const-ness
    // and trust ParMETIS to not modify the data.

    // Call ParMETIS to partition graph
    common::Timer timer1("ParMETIS: call ParMETIS_V3_PartKway");
    std::vector<idx_t> part(num_local_cells);
    assert(!part.empty());
    int err = ParMETIS_V3_PartKway(
        const_cast<idx_t*>(node_dist.data()),
        const_cast<idx_t*>(local_graph.offsets().data()),
        const_cast<idx_t*>(local_graph.array().data()), elmwgt, nullptr,
        &wgtflag, &numflag, &ncon, &nparts, tpwgts.data(), ubvec.data(),
        _options.data(), &edgecut, part.data(), &mpi_comm);
    assert(err == METIS_OK);
    timer1.stop();

    const idx_t elm_begin = node_dist[rank];
    const idx_t elm_end = node_dist[rank + 1];
    const std::int32_t ncells = elm_end - elm_begin;

    // Create a map of local nodes to their additional destination
    // processes, due to ghosting. If no ghosting, this will remain
    // empty.
    std::map<std::int32_t, std::set<std::int32_t>> local_node_to_dests;

    if (ghosting)
    {
      // Work out halo cells for current division of dual graph
      common::Timer timer2("Compute graph halo data (ParMETIS)");

      std::map<unsigned long long, std::set<std::int32_t>> halo_cell_to_remotes;

      // local indexing "i"
      for (int i = 0; i < ncells; i++)
      {
        const auto edges = local_graph.links(i);
        for (int j = 0; j < local_graph.num_links(i); ++j)
        {
          const idx_t other_cell = edges[j];
          if (other_cell < elm_begin || other_cell >= elm_end)
          {
            const int remote = std::upper_bound(node_dist.begin(),
                                                node_dist.end(), other_cell)
                               - node_dist.begin() - 1;

            assert(remote < size);
            if (halo_cell_to_remotes.find(i) == halo_cell_to_remotes.end())
              halo_cell_to_remotes[i] = std::set<std::int32_t>();
            halo_cell_to_remotes[i].insert(remote);
          }
        }
      }

      // Do halo exchange of cell partition data
      std::vector<std::vector<std::int64_t>> send_cell_partition(size);
      for (const auto& hcell : halo_cell_to_remotes)
      {
        for (auto proc : hcell.second)
        {
          assert(proc < size);

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
      const std::vector<idx_t>& adjncy = local_graph.array();

      // Generate map for where new boundary cells need to be sent
      for (std::int32_t i = 0; i < ncells; i++)
      {
        const std::int32_t proc_this = part[i];
        for (idx_t j = xadj[i]; j < xadj[i + 1]; ++j)
        {
          const idx_t other_cell = adjncy[j];
          std::int32_t proc_other;

          if (other_cell < (int)elm_begin or other_cell >= (int)elm_end)
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
    dests.reserve(ncells);
    std::vector<std::int32_t> offsets(1, 0);
    for (std::int32_t i = 0; i < ncells; ++i)
    {
      dests.push_back(part[i]);
      if (const auto it = local_node_to_dests.find(i);
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
//-----------------------------------------------------------------------------
#endif
