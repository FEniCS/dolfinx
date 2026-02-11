// Copyright (C) 2017-2025 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "dolfinx_wrappers/graph.h"
#include "dolfinx_wrappers/caster_mpi.h"
#include <array>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/ordering.h>
#include <dolfinx/graph/partition.h>
#include <dolfinx/graph/partitioners.h>
#include <dolfinx/graph/utils.h>
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
#include <vector>

namespace nb = nanobind;

namespace dolfinx_wrappers
{
void graph(nb::module_& m)
{
  dolfinx_wrappers::declare_adjacency_list_init<std::int32_t, std::nullptr_t>(
      m, "int32");
  dolfinx_wrappers::declare_adjacency_list_init<std::int64_t, std::nullptr_t>(
      m, "int64");
  dolfinx_wrappers::declare_adjacency_list<
      std::tuple<int, std::size_t, std::int8_t>,
      std::pair<std::int32_t, std::int32_t>>(m, "int_sizet_int8__int32_int32");

  using partition_fn
      = std::function<dolfinx::graph::AdjacencyList<std::int32_t>(
          MPICommWrapper, int,
          const dolfinx::graph::AdjacencyList<std::int64_t>&, bool)>;
  m.def(
      "partitioner",
      []() -> partition_fn
      {
        return dolfinx_wrappers::create_partitioner_py(
            dolfinx::graph::partition_graph);
      },
      "Default graph partitioner");

#ifdef HAS_PTSCOTCH
  m.def(
      "partitioner_scotch",
      [](double imbalance, int seed) -> partition_fn
      {
        return dolfinx_wrappers::create_partitioner_py(
            dolfinx::graph::scotch::partitioner(
                dolfinx::graph::scotch::strategy::none, imbalance, seed));
      },
      nb::arg("imbalance") = 0.025, nb::arg("seed") = 0,
      "SCOTCH graph partitioner");
#endif
#ifdef HAS_PARMETIS
  m.def(
      "partitioner_parmetis",
      [](double imbalance, std::array<int, 3> options) -> partition_fn
      {
        return dolfinx_wrappers::create_partitioner_py(
            dolfinx::graph::parmetis::partitioner(imbalance, options));
      },
      nb::arg("imbalance") = 1.02,
      nb::arg("options") = std ::array<int, 3>({1, 0, 5}),
      "ParMETIS graph partitioner");
#endif
#ifdef HAS_KAHIP
  m.def(
      "partitioner_kahip",
      [](int mode = 1, int seed = 1, double imbalance = 0.03,
         bool suppress_output = true) -> partition_fn
      {
        return dolfinx_wrappers::create_partitioner_py(
            dolfinx::graph::kahip::partitioner(mode, seed, imbalance,
                                               suppress_output));
      },
      nb::arg("mode") = 1, nb::arg("seed") = 1, nb::arg("imbalance") = 0.03,
      nb::arg("suppress_output") = true, "KaHIP graph partitioner");
#endif

  m.def("reorder_gps", &dolfinx::graph::reorder_gps, nb::arg("graph"));

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
