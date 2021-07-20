// Copyright (C) 2008-2016 Niclas Jansson, Ola Skavhaug, Anders Logg,
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
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
                   std::int32_t, bool ghosting)
  {
    LOG(INFO) << "Compute graph partition using ParMETIS";
    common::Timer timer("Compute graph partition (ParMETIS)");

    if (graph.num_nodes() == 0)
    {
      throw std::runtime_error(
          "ParMETIS cannot partition a graph where one of the MPI ranks has no "
          "data. Try PT-SCOTCH of KaHIP instead.");
    }

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
    idx_t wgtflag(0), edgecut(0), numflag(0);
    std::vector<real_t> tpwgts(ncon * nparts,
                               1.0 / static_cast<real_t>(nparts));
    std::vector<real_t> ubvec(ncon, 1.05);

    // Communicate number of nodes between all processors
    std::vector<idx_t> node_disp(size + 1, 0);
    const idx_t num_local_nodes = graph.num_nodes();
    MPI_Allgather(&num_local_nodes, 1, MPI::mpi_type<idx_t>(),
                  node_disp.data() + 1, 1, MPI::mpi_type<idx_t>(), mpi_comm);
    std::partial_sum(node_disp.begin(), node_disp.end(), node_disp.begin());

    // Call ParMETIS to partition graph
    common::Timer timer1("ParMETIS: call ParMETIS_V3_PartKway");
    std::vector<idx_t> part(num_local_nodes);
    {
      std::vector<idx_t> _array(graph.array().begin(), graph.array().end()),
          _offsets(graph.offsets().begin(), graph.offsets().end());
      int err = ParMETIS_V3_PartKway(
          node_disp.data(), _offsets.data(), _array.data(), elmwgt, nullptr,
          &wgtflag, &numflag, &ncon, &nparts, tpwgts.data(), ubvec.data(),
          _options.data(), &edgecut, part.data(), &mpi_comm);
      assert(err == METIS_OK);
    }
    timer1.stop();

    const idx_t range0 = node_disp[rank];
    const idx_t range1 = node_disp[rank + 1];
    assert(range1 - range0 == graph.num_nodes());

    // Create a map from local nodes to their additional destination
    // ranks, due to ghosting. If no ghosting, this will remain empty.
    std::map<std::int32_t, std::set<std::int32_t>> local_node_to_dests;

    if (ghosting)
    {
      // Work out halo nodes for current division of graph
      common::Timer timer2("Compute graph halo data (ParMETIS)");

      std::map<std::int32_t, std::set<int>> halo_node_to_remotes;

      // local indexing "i"
      for (int node = 0; node < graph.num_nodes(); ++node)
      {
        for (auto node1 : graph.links(node))
        {
          if (node1 < range0 or node1 >= range1)
          {
            const int remote_rank
                = std::distance(node_disp.begin(),
                                std::upper_bound(node_disp.begin(),
                                                 node_disp.end(), node1))
                  - 1;
            assert(remote_rank < size);
            halo_node_to_remotes[node].insert(remote_rank);
          }
        }
      }

      // Do halo exchange of node partition data
      std::vector<std::vector<std::int64_t>> send_node_partition(size);
      for (const auto& halo_node : halo_node_to_remotes)
      {
        const std::int32_t node_index = halo_node.first;
        std::for_each(
            halo_node.second.cbegin(), halo_node.second.cend(),
            [&send_node_partition, &part, node_index, range0](auto proc)
            {
              assert(static_cast<std::size_t>(proc)
                     < send_node_partition.size());
              // (0) global node index and (1) partitioning
              send_node_partition[proc].push_back(node_index + range0);
              send_node_partition[proc].push_back(part[node_index]);
            });
      }

      // Actual halo exchange
      const std::vector<std::int64_t> recv_node_partition
          = dolfinx::MPI::all_to_all(
                mpi_comm,
                graph::AdjacencyList<std::int64_t>(send_node_partition))
                .array();

      // Construct a map from all currently foreign nodes to their new
      // partition number
      std::map<std::int64_t, std::int32_t> node_ownership;
      for (std::size_t p = 0; p < recv_node_partition.size(); p += 2)
      {
        node_ownership.insert(
            {recv_node_partition[p], recv_node_partition[p + 1]});
      }

      // Generate map for where new boundary nodes need to be sent
      for (std::int32_t node = 0; node < graph.num_nodes(); ++node)
      {
        const std::int32_t proc_this = part[node];
        for (auto node1 : graph.links(node))
        {
          std::int32_t proc_other;
          if (node1 < range0 or node1 >= range1)
          {
            const auto find_other_proc = node_ownership.find(node1);
            // remote cell - should be in map
            assert(find_other_proc != node_ownership.end());
            proc_other = find_other_proc->second;
          }
          else
            proc_other = part[node1 - range0];

          if (proc_this != proc_other)
            local_node_to_dests[node].insert(proc_other);
        }
      }
      timer2.stop();
    }

    // Convert to offset format for AdjacencyList
    std::vector<std::int32_t> dests, offsets(1, 0);
    dests.reserve(graph.num_nodes());
    offsets.reserve(graph.num_nodes() + 1);
    for (std::int32_t node = 0; node < graph.num_nodes(); ++node)
    {
      dests.push_back(part[node]);
      if (const auto it = local_node_to_dests.find(node);
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
