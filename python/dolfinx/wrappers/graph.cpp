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
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <optional>
#include <ranges>
#include <span>
#include <vector>

namespace nb = nanobind;

namespace
{
/// Convert an optional span of weights to the optional ndarray a Python
/// partitioner expects. std::nullopt (no weights) becomes Python None,
/// not a zero-length array.
std::optional<nb::ndarray<const std::int32_t, nb::numpy>>
weights_to_ndarray(std::optional<std::span<const std::int32_t>> w)
{
  if (!w)
    return std::nullopt;
  return nb::ndarray<const std::int32_t, nb::numpy>(
      w->data(), std::initializer_list<std::size_t>({w->size()}));
}

/// Inverse of weights_to_ndarray: Python None becomes std::nullopt, not
/// a zero-length span.
std::optional<std::span<const std::int32_t>> weights_to_span(
    std::optional<nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>> w)
{
  if (!w)
    return std::nullopt;
  return std::span<const std::int32_t>(w->data(), w->size());
}
} // namespace

namespace dolfinx_wrappers
{
dolfinx::graph::partition_fn
partitioner_wrap_py_to_cpp(const PythonPartitionFunction& p)
{
  return [p](MPI_Comm comm, int nparts,
             const dolfinx::graph::AdjacencyList<std::int64_t>& local_graph,
             std::optional<std::span<const std::int32_t>> node_weights,
             std::optional<std::span<const std::int32_t>> edge_weights,
             bool ghosting)
  {
    return p(MPICommWrapper(comm), nparts, local_graph,
             weights_to_ndarray(node_weights), weights_to_ndarray(edge_weights),
             ghosting);
  };
}

dolfinx::graph::geom_partition_fn
partitioner_wrap_py_to_cpp(const PythonGeoPartitionFunction& p)
{
  return [p](MPI_Comm comm, int nparts, std::span<const double> x, int gdim,
             std::optional<std::span<const std::int32_t>> node_weights)
             -> std::vector<int>
  {
    std::size_t shape0 = gdim > 0 ? x.size() / gdim : 0;
    std::size_t shape[2] = {shape0, static_cast<std::size_t>(gdim)};
    nb::ndarray<const double, nb::ndim<2>, nb::numpy> x_nb(x.data(), 2, shape,
                                                           nb::handle());
    nb::ndarray<const int, nb::ndim<1>, nb::numpy> dest = p(
        MPICommWrapper(comm), nparts, x_nb, weights_to_ndarray(node_weights));
    return std::vector<int>(dest.data(), dest.data() + dest.size());
  };
}

dolfinx::graph::hybrid_partition_fn
partitioner_wrap_py_to_cpp(const PythonHybridPartitionFunction& p)
{
  return [p](MPI_Comm comm, int nparts,
             const dolfinx::graph::AdjacencyList<std::int64_t>& local_graph,
             std::span<const double> x,
             std::optional<std::span<const std::int32_t>> node_weights,
             std::optional<std::span<const std::int32_t>> edge_weights,
             bool ghosting)
  {
    std::size_t num_nodes = static_cast<std::size_t>(local_graph.num_nodes());
    std::size_t gdim = num_nodes > 0 ? x.size() / num_nodes : 0;
    std::size_t shape[2] = {num_nodes, gdim};
    nb::ndarray<const double, nb::ndim<2>, nb::numpy> x_nb(x.data(), 2, shape,
                                                           nb::handle());
    return p(MPICommWrapper(comm), nparts, local_graph, x_nb,
             weights_to_ndarray(node_weights), weights_to_ndarray(edge_weights),
             ghosting);
  };
}

void graph(nb::module_& m)
{
  declare_adjacency_list_init<std::int32_t, std::nullptr_t>(m, "int32");
  declare_adjacency_list_init<std::int64_t, std::nullptr_t>(m, "int64");
  declare_adjacency_list<std::tuple<int, std::size_t, std::int8_t>,
                         std::pair<std::int32_t, std::int32_t>>(
      m, "int_sizet_int8__int32_int32");

  m.def(
      "partitioner", []() -> GraphPartitioner
      { return GraphPartitioner{dolfinx::graph::partition_graph}; },
      "Default graph partitioner. Returns a GraphPartitioner, ready to "
      "pass directly as create_mesh's partitioner argument.");

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
         dolfinx::graph::scotch::strategy strategy) -> GraphPartitioner
      {
        return GraphPartitioner{
            dolfinx::graph::scotch::partitioner(strategy, imbalance, seed)};
      },
      nb::arg("imbalance") = 0.025, nb::arg("seed") = 0,
      nb::arg("strategy") = dolfinx::graph::scotch::strategy::speed,
      "SCOTCH graph partitioner. Returns a GraphPartitioner, ready to pass "
      "directly as create_mesh's partitioner argument.");
#endif
#ifdef HAS_PARMETIS
  m.def(
      "partitioner_parmetis",
      [](double imbalance, std::array<int, 3> options) -> GraphPartitioner
      {
        return GraphPartitioner{
            dolfinx::graph::parmetis::partitioner(imbalance, options)};
      },
      nb::arg("imbalance") = 1.02,
      nb::arg("options") = std::array<int, 3>({1, 0, 5}),
      "ParMETIS graph partitioner. Returns a GraphPartitioner, ready to "
      "pass directly as create_mesh's partitioner argument.");

  m.def(
      "partitioner_parmetis_hybrid",
      [](double imbalance, std::array<int, 3> options) -> HybridPartitioner
      {
        return HybridPartitioner{
            dolfinx::graph::parmetis::geom_partitioner_kway(imbalance,
                                                            options)};
      },
      nb::arg("imbalance") = 1.02,
      nb::arg("options") = std::array<int, 3>({1, 0, 5}),
      "ParMETIS geometric (space-filling curve redistribution followed by "
      "k-way) graph partitioner. Returns a HybridPartitioner, ready to pass "
      "directly as create_mesh's partitioner argument.");
#endif
#ifdef HAS_KAHIP
  m.def(
      "partitioner_kahip",
      [](int mode = 1, int seed = 1, double imbalance = 0.03,
         bool suppress_output = true) -> GraphPartitioner
      {
        return GraphPartitioner{dolfinx::graph::kahip::partitioner(
            mode, seed, imbalance, suppress_output)};
      },
      nb::arg("mode") = 1, nb::arg("seed") = 1, nb::arg("imbalance") = 0.03,
      nb::arg("suppress_output") = true,
      "KaHIP graph partitioner. Returns a GraphPartitioner, ready to pass "
      "directly as create_mesh's partitioner argument.");
#endif

  // Opaque holders for partition_fn, geom_partition_fn and
  // hybrid_partition_fn. create_mesh recognises these types -- distinct
  // from a plain callable and from each other -- to route them to the
  // matching AnyPartitionFunction alternative, which computes cell
  // centroids itself for the geometric/hybrid cases. __call__ is
  // exposed on all three for direct low-level use; the geometric and
  // hybrid __call__ take cell centroids, not raw node coordinates.
  nb::class_<GraphPartitioner>(
      m, "GraphPartitioner",
      "Cell partitioner using the mesh dual graph, e.g. as returned by "
      "partitioner, partitioner_scotch, partitioner_parmetis or "
      "partitioner_kahip. Pass it as create_mesh's partitioner argument; "
      "it is also directly callable for low-level use.")
      .def(
          "__call__",
          [](const GraphPartitioner& self, MPICommWrapper comm, int nparts,
             const dolfinx::graph::AdjacencyList<std::int64_t>& local_graph,
             std::optional<
                 nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>>
                 node_weights,
             std::optional<
                 nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>>
                 edge_weights,
             bool ghosting)
          {
            return self.fn(comm.get(), nparts, local_graph,
                           weights_to_span(node_weights),
                           weights_to_span(edge_weights), ghosting);
          },
          nb::arg("comm"), nb::arg("nparts"), nb::arg("local_graph"),
          nb::arg("node_weights").none(), nb::arg("edge_weights").none(),
          nb::arg("ghosting"),
          "Compute the destination rank for each node of local_graph.");

  nb::class_<GeometricPartitioner>(
      m, "GeometricPartitioner",
      "Cell partitioner using cell positions rather than the mesh dual "
      "graph, e.g. create_geometric_cell_partitioner's result, or the "
      "partition_morton, partition_hilbert and partitioner_parmetis_geom "
      "built-ins. Pass it as create_mesh's partitioner argument; it is "
      "also directly callable for low-level use, e.g. via "
      "compute_cell_centroids.")
      .def(
          "__call__",
          [](const GeometricPartitioner& self, MPICommWrapper comm, int nparts,
             nb::ndarray<const double, nb::ndim<2>, nb::c_contig> x,
             std::optional<
                 nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>>
                 node_weights)
          {
            int gdim = static_cast<int>(x.shape(1));
            return as_nbarray(self.fn(
                comm.get(), nparts, std::span<const double>(x.data(), x.size()),
                gdim, weights_to_span(node_weights)));
          },
          nb::arg("comm"), nb::arg("nparts"), nb::arg("x").noconvert(),
          nb::arg("node_weights").none(),
          "Compute the destination rank for each row of x (cell centroids, "
          "not raw node coordinates -- see compute_cell_centroids).");

  // partition_morton, partition_hilbert and partitioner_parmetis_geom
  // take no configuration, so they are plain GeometricPartitioner
  // attributes rather than factory functions.
  m.attr("partition_morton")
      = GeometricPartitioner{dolfinx::graph::partition_sfc_morton};
  m.attr("partition_hilbert")
      = GeometricPartitioner{dolfinx::graph::partition_sfc_hilbert};
#ifdef HAS_PARMETIS
  m.attr("partitioner_parmetis_geom")
      = GeometricPartitioner{dolfinx::graph::parmetis::geom_partitioner};
#endif

  nb::class_<HybridPartitioner>(
      m, "HybridPartitioner",
      "Cell partitioner returned by create_hybrid_cell_partitioner. Pass "
      "it as create_mesh's partitioner argument; it is also directly "
      "callable for low-level use, e.g. via compute_cell_centroids.")
      .def(
          "__call__",
          [](const HybridPartitioner& self, MPICommWrapper comm, int nparts,
             const dolfinx::graph::AdjacencyList<std::int64_t>& dual_graph,
             nb::ndarray<const double, nb::ndim<2>, nb::c_contig> x,
             std::optional<
                 nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>>
                 node_weights,
             std::optional<
                 nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>>
                 edge_weights,
             bool ghosting)
          {
            return self.fn(comm.get(), nparts, dual_graph,
                           std::span<const double>(x.data(), x.size()),
                           weights_to_span(node_weights),
                           weights_to_span(edge_weights), ghosting);
          },
          nb::arg("comm"), nb::arg("nparts"), nb::arg("dual_graph"),
          nb::arg("x").noconvert(), nb::arg("node_weights").none(),
          nb::arg("edge_weights").none(), nb::arg("ghosting"),
          "Compute the destination rank for each cell, given the mesh "
          "dual graph and cell centroids (not raw node coordinates -- "
          "see compute_cell_centroids).");

  m.def(
      "create_geometric_cell_partitioner",
      [](const PythonGeoPartitionFunction& part) -> GeometricPartitioner
      { return GeometricPartitioner{partitioner_wrap_py_to_cpp(part)}; },
      nb::arg("part"),
      "Create a geometric cell partitioner from a geometric graph "
      "partitioning function. Pass the result as create_mesh's "
      "partitioner argument; create_mesh supplies the cell centroids "
      "itself from whatever coordinate data it is given. The resulting "
      "partitioner never ghosts: it has no cell topology, so it cannot "
      "build the mesh dual graph. Use create_hybrid_cell_partitioner for "
      "a partitioner that also needs the dual graph.");

  m.def(
      "create_hybrid_cell_partitioner",
      [](const PythonHybridPartitionFunction& part) -> HybridPartitioner
      { return HybridPartitioner{partitioner_wrap_py_to_cpp(part)}; },
      nb::arg("part"),
      "Create a cell partitioner from a hybrid graph partitioning "
      "function that needs both the mesh dual graph and cell positions. "
      "Unlike create_geometric_cell_partitioner, the dual graph is "
      "always supplied, regardless of ghost_mode. Pass the result as "
      "create_mesh's partitioner argument.");

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
