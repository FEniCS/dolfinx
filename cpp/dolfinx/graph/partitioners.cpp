// Copyright (C) 2019-2020 Garth N. Wells, Chris Richardson and Igor A. Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "partitioners.h"
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <map>
#include <set>
#include <vector>

#ifdef HAS_PARMETIS
#include <parmetis.h>
#endif

#ifdef HAS_KAHIP
#include <parhip_interface.h>
#endif

using namespace dolfinx;

namespace
{
//-----------------------------------------------------------------------------

/// Build ParMETIS adjacency list data
template <typename T>
std::array<std::vector<T>, 3>
build_adjacency_data(MPI_Comm comm,
                     const graph::AdjacencyList<std::int64_t>& graph)
{
  // Communicate number of nodes between all processors
  const int size = dolfinx::MPI::size(comm);
  std::vector<T> node_disp(size + 1, 0);
  const T num_local_nodes = graph.num_nodes();
  MPI_Allgather(&num_local_nodes, 1, dolfinx::MPI::mpi_type<T>(),
                node_disp.data() + 1, 1, dolfinx::MPI::mpi_type<T>(), comm);
  std::partial_sum(node_disp.begin(), node_disp.end(), node_disp.begin());
  return {std::move(node_disp),
          std::vector<T>(graph.array().begin(), graph.array().end()),
          std::vector<T>(graph.offsets().begin(), graph.offsets().end())};
}
//-----------------------------------------------------------------------------
template <typename T>
graph::AdjacencyList<std::int32_t> compute_destination_ranks(
    MPI_Comm comm, const graph::AdjacencyList<std::int64_t>& graph,
    const std::vector<T>& node_disp, const std::vector<T>& part)
{
  // Create a map from local nodes to their additional destination
  // ranks, due to ghosting.

  // Work out halo nodes for current division of graph
  common::Timer timer2("Compute graph halo data (ParMETIS/KaHIP)");

  const int rank = dolfinx::MPI::rank(comm);
  const int size = dolfinx::MPI::size(comm);
  const std::int64_t range0 = node_disp[rank];
  const std::int64_t range1 = node_disp[rank + 1];
  assert(static_cast<std::int32_t>(range1 - range0) == graph.num_nodes());

  // local indexing "i"
  std::map<std::int32_t, std::set<int>> halo_node_to_remotes;
  for (int node = 0; node < graph.num_nodes(); ++node)
  {
    for (auto node1 : graph.links(node))
    {
      if (node1 < range0 or node1 >= range1)
      {
        const int remote_rank
            = std::distance(
                  node_disp.begin(),
                  std::upper_bound(node_disp.begin(), node_disp.end(), node1))
              - 1;
        assert(remote_rank < size);
        halo_node_to_remotes[node].insert(remote_rank);
      }
    }
  }

  // Loop over each halo node and count number of values to be sent to
  // each rank
  std::vector<std::int32_t> count(size, 0);
  for (const auto& halo_node : halo_node_to_remotes)
  {
    for (auto rank : halo_node.second)
      count[rank] += 2;
  }

  // Compute displacement (offsets)
  std::vector<std::int32_t> disp(size + 1, 0);
  std::partial_sum(count.begin(), count.end(), std::next(disp.begin()));

  // Pack send data for exchanging node partition data
  graph::AdjacencyList<std::int64_t> send_node_partition(
      std::vector<std::int64_t>(disp.back()), disp);
  std::vector<std::int32_t> pos(size, 0);
  for (const auto& halo_node : halo_node_to_remotes)
  {
    const std::int32_t node_index = halo_node.first;
    std::for_each(
        halo_node.second.cbegin(), halo_node.second.cend(),
        [&send_node_partition, &pos, &part, node_index, range0](auto rank)
        {
          assert(rank < send_node_partition.num_nodes());
          // (0) global node index and (1) partitioning
          auto dests = send_node_partition.links(rank);
          dests[pos[rank]++] = node_index + range0;
          dests[pos[rank]++] = part[node_index];
        });
  }

  // Do halo exchange
  const std::vector<std::int64_t> recv_node_partition
      = dolfinx::MPI::all_to_all(comm, send_node_partition).array();

  // Construct a map from all currently foreign nodes to their new
  // partition number
  std::map<std::int64_t, std::int32_t> node_ownership;
  for (std::size_t p = 0; p < recv_node_partition.size(); p += 2)
    node_ownership.insert({recv_node_partition[p], recv_node_partition[p + 1]});

  // Generate map for where new boundary nodes need to be sent
  std::map<std::int32_t, std::set<std::int32_t>> local_node_to_dests;
  for (std::int32_t node0 = 0; node0 < graph.num_nodes(); ++node0)
  {
    const std::int32_t node0_rank = part[node0];
    for (auto node1 : graph.links(node0))
    {
      std::int32_t node1_rank;
      if (node1 < range0 or node1 >= range1)
      {
        const auto it = node_ownership.find(node1);
        // remote cell - should be in map
        assert(it != node_ownership.end());
        node1_rank = it->second;
      }
      else
        node1_rank = part[node1 - range0];

      if (node0_rank != node1_rank)
        local_node_to_dests[node0].insert(node1_rank);
    }
  }

  // Convert to  AdjacencyList
  std::vector<std::int32_t> dests, offsets(1, 0);
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
}

//-----------------------------------------------------------------------------
#ifdef HAS_PARMETIS
template <typename T>
std::vector<int> adaptive_repartition(MPI_Comm comm,
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
  idx_t nparts = dolfinx::MPI::size(comm);

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
      &edgecut, part.data(), &comm);
  assert(err == METIS_OK);
  timer1.stop();

  // Copy cell partition data and return
  return std::vector<int>(part.begin(), part.end());
}
//-----------------------------------------------------------------------------
template <typename T>
std::vector<int> refine(MPI_Comm comm, const graph::AdjacencyList<T>& adj_graph)
{
  common::Timer timer("Compute graph partition (ParMETIS Refine)");

  // Get some MPI data
  const int process_number = dolfinx::MPI::rank(comm);

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
  idx_t nparts = dolfinx::MPI::size(comm);
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
      &comm);
  assert(err == METIS_OK);
  timer1.stop();

  // Copy cell partition data
  return std::vector<int>(part.begin(), part.end());
  //-----------------------------------------------------------------------------
}
#endif
} // namespace

//-----------------------------------------------------------------------------
#ifdef HAS_PARMETIS
graph::partition_fn graph::parmetis::partitioner(double imbalance,
                                                 std::array<int, 3> options)
{
  return [imbalance, options](MPI_Comm comm, idx_t nparts,
                              const graph::AdjacencyList<std::int64_t>& graph,
                              std::int32_t, bool ghosting)
  {
    LOG(INFO) << "Compute graph partition using ParMETIS";
    common::Timer timer("Compute graph partition (ParMETIS)");

    if (graph.num_nodes() == 0)
    {
      throw std::runtime_error(
          "ParMETIS cannot partition a graph where one of the MPI ranks has no "
          "data. Try PT-SCOTCH or KaHIP instead.");
    }

    // Options for ParMETIS
    std::array<idx_t, 3> _options = {options[0], options[1], options[2]};

    // Data for  ParMETIS
    idx_t ncon = 1;
    idx_t* elmwgt = nullptr;
    idx_t wgtflag(0), edgecut(0), numflag(0);
    std::vector<real_t> tpwgts(ncon * nparts,
                               1.0 / static_cast<real_t>(nparts));
    std::array<real_t, 1> ubvec = {static_cast<real_t>(imbalance)};

    // Build adjacency list data
    common::Timer timer1("ParMETIS: build adjacency data");
    auto [node_disp, array, _offsets]
        = build_adjacency_data<idx_t>(comm, graph);
    timer1.stop();

    // Call ParMETIS to partition graph
    common::Timer timer2("ParMETIS: call ParMETIS_V3_PartKway");
    std::vector<idx_t> part(graph.num_nodes());
    int err = ParMETIS_V3_PartKway(
        node_disp.data(), _offsets.data(), array.data(), elmwgt, nullptr,
        &wgtflag, &numflag, &ncon, &nparts, tpwgts.data(), ubvec.data(),
        _options.data(), &edgecut, part.data(), &comm);
    assert(err == METIS_OK);
    timer2.stop();

    if (ghosting)
      return compute_destination_ranks(comm, graph, node_disp, part);
    else
    {
      return build_adjacency_list<std::int32_t>(
          std::vector<std::int32_t>(part.begin(), part.end()), 1);
    }
  };
}
//-----------------------------------------------------------------------------
#endif

#ifdef HAS_KAHIP

//----------------------------------------------------------------------------
std::function<graph::AdjacencyList<std::int32_t>(
    MPI_Comm, int, const graph::AdjacencyList<std::int64_t>&, std::int32_t,
    bool)>
graph::kahip::partitioner(int mode, int seed, double imbalance,
                          bool suppress_output)
{
  return [mode, seed, imbalance,
          suppress_output](MPI_Comm comm, int nparts,
                           const graph::AdjacencyList<std::int64_t>& graph,
                           std::int32_t, bool ghosting)
  {
    LOG(INFO) << "Compute graph partition using (parallel) KaHIP";

    common::Timer timer("Compute graph partition (KaHIP)");

    // Graph does not have vertex or adjacency weights, so we use null
    // pointers as arguments
    unsigned long long *vwgt(nullptr), *adjcwgt(nullptr);

    // Build adjacency list data
    common::Timer timer1("KaHIP: build adjacency data");
    auto [node_disp, array, offsets]
        = build_adjacency_data<unsigned long long>(comm, graph);
    timer1.stop();

    // Call KaHIP to partition graph
    common::Timer timer2("KaHIP: call ParHIPPartitionKWay");
    std::vector<unsigned long long> part(graph.num_nodes());
    int edgecut = 0;
    double _imbalance = imbalance;
    ParHIPPartitionKWay(node_disp.data(), offsets.data(), array.data(), vwgt,
                        adjcwgt, &nparts, &_imbalance, suppress_output, seed,
                        mode, &edgecut, part.data(), &comm);
    timer2.stop();

    if (ghosting)
      return compute_destination_ranks(comm, graph, node_disp, part);
    else
    {
      return build_adjacency_list<std::int32_t>(
          std::vector<std::int32_t>(part.begin(), part.end()), 1);
    }
  };
}
//----------------------------------------------------------------------------
#endif
