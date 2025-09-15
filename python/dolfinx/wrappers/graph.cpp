// Copyright (C) 2017-2025 Chris N. Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

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

namespace
{
/// Wrap a C++ graph partitioning function as a Python-ready function
template <typename Functor>
auto create_partitioner_py(Functor&& p_cpp)
{
  return [p_cpp](dolfinx_wrappers::MPICommWrapper comm, int nparts,
                 const dolfinx::graph::AdjacencyList<std::int64_t>& local_graph,
                 bool ghosting)
  { return p_cpp(comm.get(), nparts, local_graph, ghosting); };
}

template <typename T, typename U>
void declare_adjacency_list_init(nb::module_& m, std::string type)
{
  std::string pyclass_name = std::string("AdjacencyList_") + type;
  nb::class_<dolfinx::graph::AdjacencyList<T, U>>(m, pyclass_name.c_str(),
                                                  "Adjacency List")
      .def(
          "__init__",
          [](dolfinx::graph::AdjacencyList<T, U>* a,
             nb::ndarray<const T, nb::ndim<1>, nb::c_contig> adj)
          {
            std::vector<T> data(adj.data(), adj.data() + adj.size());
            new (a) dolfinx::graph::AdjacencyList<T, U>(
                dolfinx::graph::regular_adjacency_list<U>(std::move(data), 1));
          },
          nb::arg("adj").noconvert())
      .def(
          "__init__",
          [](dolfinx::graph::AdjacencyList<T, U>* a,
             nb::ndarray<const T, nb::ndim<2>, nb::c_contig> adj)
          {
            std::vector<T> data(adj.data(), adj.data() + adj.size());
            new (a) dolfinx::graph::AdjacencyList<T, U>(
                dolfinx::graph::regular_adjacency_list<U>(std::move(data),
                                                          adj.shape(1)));
          },
          nb::arg("adj").noconvert())
      .def(
          "__init__",
          [](dolfinx::graph::AdjacencyList<T, U>* a,
             nb::ndarray<const T, nb::ndim<1>, nb::c_contig> array,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> displ)
          {
            std::vector<T> data(array.data(), array.data() + array.size());
            std::vector<std::int32_t> offsets(displ.data(),
                                              displ.data() + displ.size());
            new (a) dolfinx::graph::AdjacencyList<T, U>(std::move(data),
                                                        std::move(offsets));
          },
          nb::arg("data").noconvert(), nb::arg("offsets"))
      .def(
          "links",
          [](const dolfinx::graph::AdjacencyList<T, U>& self, int i)
          {
            std::span<const T> link = self.links(i);
            return nb::ndarray<const T, nb::numpy>(link.data(), {link.size()});
          },
          nb::rv_policy::reference_internal, nb::arg("i"),
          "Links (edges) of a node")
      .def_prop_ro(
          "array",
          [](const dolfinx::graph::AdjacencyList<T, U>& self)
          {
            return nb::ndarray<const T, nb::numpy>(self.array().data(),
                                                   {self.array().size()});
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "offsets",
          [](const dolfinx::graph::AdjacencyList<T, U>& self)
          {
            return nb::ndarray<const std::int32_t, nb::numpy>(
                self.offsets().data(), {self.offsets().size()});
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro("num_nodes", &dolfinx::graph::AdjacencyList<T, U>::num_nodes)
      .def("__eq__", &dolfinx::graph::AdjacencyList<T, U>::operator==,
           nb::is_operator())
      .def("__repr__", &dolfinx::graph::AdjacencyList<T, U>::str)
      .def("__len__", &dolfinx::graph::AdjacencyList<T, U>::num_nodes);
}

template <typename T, typename U>
void declare_adjacency_list(nb::module_& m, std::string type)
{
  std::string pyclass_name = std::string("AdjacencyList_") + type;
  nb::class_<dolfinx::graph::AdjacencyList<T, U>>(m, pyclass_name.c_str(),
                                                  "Adjacency List")
      .def_prop_ro(
          "offsets",
          [](const dolfinx::graph::AdjacencyList<T, U>& self)
          {
            return nb::ndarray<const std::int32_t, nb::numpy>(
                self.offsets().data(), {self.offsets().size()});
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro("num_nodes", &dolfinx::graph::AdjacencyList<T, U>::num_nodes)
      .def("__eq__", &dolfinx::graph::AdjacencyList<T, U>::operator==,
           nb::is_operator())
      .def("__len__", &dolfinx::graph::AdjacencyList<T, U>::num_nodes);
}
} // namespace

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
          const dolfinx::graph::AdjacencyList<std::int64_t>&, bool)>;
  m.def(
      "partitioner", []() -> partition_fn
      { return create_partitioner_py(dolfinx::graph::partition_graph); },
      "Default graph partitioner");

#ifdef HAS_PTSCOTCH
  m.def(
      "partitioner_scotch",
      [](double imbalance, int seed) -> partition_fn
      {
        return create_partitioner_py(dolfinx::graph::scotch::partitioner(
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
        return create_partitioner_py(
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
        return create_partitioner_py(dolfinx::graph::kahip::partitioner(
            mode, seed, imbalance, suppress_output));
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
