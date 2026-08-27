// Copyright (C) 2017-2026 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "dolfinx_wrappers/graph.h"
#include "dolfinx_wrappers/array.h"
#include "dolfinx_wrappers/caster_mpi.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/ordering.h>
#include <dolfinx/graph/partition.h>
#include <dolfinx/graph/partitioners.h>
#include <dolfinx/graph/sfc.h>
#include <dolfinx/graph/utils.h>
#include <functional>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <ranges>
#include <span>
#include <vector>

namespace nb = nanobind;

namespace dolfinx_wrappers
{
void graph(nb::module_& m)
{
  declare_adjacency_list_init<std::int32_t, std::nullptr_t>(m, "int32");
  declare_adjacency_list_init<std::int64_t, std::nullptr_t>(m, "int64");
  declare_adjacency_list<std::tuple<int, std::size_t, std::int8_t>,
                         std::pair<std::int32_t, std::int32_t>>(
      m, "int_sizet_int8__int32_int32");

  using partition_fn
      = std::function<dolfinx::graph::AdjacencyList<std::int32_t>(
          MPICommWrapper, int,
          const dolfinx::graph::AdjacencyList<std::int64_t>&,
          nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>,
          nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>, bool)>;
  using geom_partition_fn
      = std::function<dolfinx::graph::AdjacencyList<std::int32_t>(
          MPICommWrapper, int,
          nb::ndarray<const double, nb::ndim<2>, nb::c_contig>)>;
  m.def(
      "partitioner", []() -> partition_fn
      { return create_partitioner_py(dolfinx::graph::partition_graph); },
      "Default graph partitioner");

  nb::enum_<dolfinx::graph::sfc::curve>(m, "SFCCurve")
      .value("morton", dolfinx::graph::sfc::curve::morton)
      .value("hilbert", dolfinx::graph::sfc::curve::hilbert);

  m.def(
      "geom_partitioner_sfc",
      [](dolfinx::graph::sfc::curve curve) -> geom_partition_fn
      {
        return create_geom_partitioner_py(
            dolfinx::graph::sfc::partitioner(curve));
      },
      nb::arg("curve"), "Space-filling curve geometric graph partitioner");

#ifdef HAS_PTSCOTCH
  nb::enum_<dolfinx::graph::scotch::strategy>(m, "SCOTCHStrategy")
      .value("none", dolfinx::graph::scotch::strategy::none)
      .value("balance", dolfinx::graph::scotch::strategy::balance)
      .value("quality", dolfinx::graph::scotch::strategy::quality)
      .value("safety", dolfinx::graph::scotch::strategy::safety)
      .value("speed", dolfinx::graph::scotch::strategy::speed)
      .value("scalability", dolfinx::graph::scotch::strategy::scalability);

  m.def(
      "partitioner_scotch",
      [](double imbalance, int seed,
         dolfinx::graph::scotch::strategy strategy) -> partition_fn
      {
        return create_partitioner_py(
            dolfinx::graph::scotch::partitioner(strategy, imbalance, seed));
      },
      nb::arg("imbalance") = 0.025, nb::arg("seed") = 0,
      nb::arg("strategy") = dolfinx::graph::scotch::strategy::speed,
      "SCOTCH graph partitioner");
#endif
#ifdef HAS_PARMETIS
  m.def(
      "partitioner_parmetis",
      [](double imbalance, std::array<int, 3> options) -> partition_fn
      {
        return create_partitioner_py(
            dolfinx::graph::parmetis::partitioner(imbalance, options));
      },
      nb::arg("imbalance") = 1.02,
      nb::arg("options") = std::array<int, 3>({1, 0, 5}),
      "ParMETIS graph partitioner");

  m.def(
      "geom_partitioner_parmetis",
      []() -> geom_partition_fn
      {
        return create_geom_partitioner_py(
            dolfinx::graph::parmetis::geom_partitioner());
      },
      "ParMETIS geometric (space-filling curve) graph partitioner");

  using hybrid_partition_fn
      = std::function<dolfinx::graph::AdjacencyList<std::int32_t>(
          MPICommWrapper, int,
          const dolfinx::graph::AdjacencyList<std::int64_t>&,
          nb::ndarray<const double, nb::ndim<2>, nb::c_contig>, bool)>;
  m.def(
      "geom_partitioner_parmetis_kway",
      [](double imbalance, std::array<int, 3> options) -> hybrid_partition_fn
      {
        return create_hybrid_partitioner_py(
            dolfinx::graph::parmetis::geom_partitioner_kway(imbalance,
                                                            options));
      },
      nb::arg("imbalance") = 1.02,
      nb::arg("options") = std::array<int, 3>({1, 0, 5}),
      "ParMETIS geometric (space-filling curve redistribution followed by "
      "k-way) graph partitioner");
#endif
#ifdef HAS_KAHIP
  m.def(
      "partitioner_kahip",
      [](int mode = 1, int seed = 1, double imbalance = 0.03,
         bool suppress_output = true) -> partition_fn
      {
        return create_partitioner_py(dolfinx::graph::kahip::partitioner(
            mode, seed, imbalance, suppress_output));
      },
      nb::arg("mode") = 1, nb::arg("seed") = 1, nb::arg("imbalance") = 0.03,
      nb::arg("suppress_output") = true, "KaHIP graph partitioner");
#endif

  m.def("reorder_rcm", &dolfinx::graph::reorder_rcm, nb::arg("graph"));

  m.def(
      "distribute",
      [](const MPICommWrapper comm,
         nb::ndarray<const std::int64_t, nb::ndim<2>, nb::c_contig> list,
         const dolfinx::graph::AdjacencyList<std::int32_t>& destinations)
      {
        std::size_t shape1 = list.shape(1);
        auto [recv, src, orig_idx, ghost_owners]
            = dolfinx::graph::build::distribute(
                comm.get(),
                std::span<const std::int64_t>(list.data(), list.size()),
                {static_cast<std::size_t>(list.shape(0)), shape1},
                destinations);
        std::size_t num_recv = shape1 > 0 ? recv.size() / shape1 : 0;
        return std::make_tuple(as_nbarray(std::move(recv), {num_recv, shape1}),
                               as_nbarray(std::move(src)),
                               as_nbarray(std::move(orig_idx)),
                               as_nbarray(std::move(ghost_owners)));
      },
      nb::arg("comm"), nb::arg("list"), nb::arg("destinations"),
      "Distribute rows of a fixed-degree array (e.g. mesh cells, shape "
      "(num_nodes, degree)) to destination ranks, using a scalable "
      "neighbourhood exchange. `destinations` gives the destination "
      "rank(s) for each row (the owner first, then any ghost "
      "destinations), as returned by a cell/graph partitioner. Returns "
      "(received rows, source rank per received row, original global "
      "index per received row, owning rank of ghost rows).");

  m.def(
      "comm_graph", [](const dolfinx::common::IndexMap& map, int root)
      { return dolfinx::graph::comm_graph(map, root); }, nb::arg("map"),
      nb::arg("root") = 0,
      "Build a graph representing parallel communication patterns.");

  m.def(
      "comm_graph_data",
      [](dolfinx::graph::AdjacencyList<
          std::tuple<int, std::size_t, std::int8_t>,
          std::pair<std::int32_t, std::int32_t>>& g)
      {
        std::vector<std::tuple<int, int, std::map<std::string, std::size_t>>>
            adj;
        for (std::int32_t n = 0; n < g.num_nodes(); ++n)
        {
          for (auto [e, w, local] : g.links(n))
          {
            adj.emplace_back(n, e,
                             std::map<std::string, std::size_t>{
                                 {"local", local}, {"weight", w}});
          }
        }

        std::vector<
            std::pair<std::int32_t, std::map<std::string, std::int32_t>>>
            nodes;
        std::ranges::transform(
            g.node_data().value(), std::ranges::views::iota(0),
            std::back_inserter(nodes),
            [](auto data, auto n)
            {
              return std::pair(
                  n, std::map<std::string, std::int32_t>{
                         {"weight", data.first}, {"num_ghosts", data.second}});
            });

        return std::pair(std::move(adj), std::move(nodes));
      },
      "Build a graph edge and node data representing parallel communication "
      "patterns. Can be used to creat NetworkX graphs.");

  m.def(
      "comm_to_json",
      [](dolfinx::graph::AdjacencyList<
          std::tuple<int, std::size_t, std::int8_t>,
          std::pair<std::int32_t, std::int32_t>>& g)
      { return dolfinx::graph::comm_to_json(g); },
      "Build a JSON string representation of a parallel communication "
      "graph that can use used by build a NetworkX graph.");
}
} // namespace dolfinx_wrappers
