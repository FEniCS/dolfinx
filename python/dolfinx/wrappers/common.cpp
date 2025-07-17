// Copyright (C) 2017-2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "dolfinx_wrappers/MPICommWrapper.h"
#include "dolfinx_wrappers/array.h"
#include "dolfinx_wrappers/caster_mpi.h"
#include <complex>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/Scatterer.h>
#include <dolfinx/common/Table.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/defines.h>
#include <dolfinx/common/log.h>
#include <dolfinx/common/timing.h>
#include <dolfinx/common/utils.h>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/chrono.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace dolfinx_wrappers
{

/// Return true if DOLFINx is compiled with petsc4py
consteval bool has_petsc4py()
{
#ifdef HAS_PETSC4PY
  return true;
#else
  return false;
#endif
}

template <typename T>
void add_scatter_functions(nb::class_<dolfinx::common::Scatterer<>>& sc)
{
  sc.def(
      "scatter_fwd",
      [](dolfinx::common::Scatterer<>& self,
         nb::ndarray<const T, nb::ndim<1>, nb::c_contig> local_data,
         nb::ndarray<T, nb::ndim<1>, nb::c_contig> remote_data)
      {
        self.scatter_fwd(std::span(local_data.data(), local_data.size()),
                         std::span(remote_data.data(), remote_data.size()));
      },
      nb::arg("local_data"), nb::arg("remote_data"));

  sc.def(
      "scatter_rev",
      [](dolfinx::common::Scatterer<>& self,
         nb::ndarray<T, nb::ndim<1>, nb::c_contig> local_data,
         nb::ndarray<const T, nb::ndim<1>, nb::c_contig> remote_data)
      {
        self.scatter_rev(std::span(local_data.data(), local_data.size()),
                         std::span(remote_data.data(), remote_data.size()),
                         std::plus<T>());
      },
      nb::arg("local_data"), nb::arg("remote_data"));
}

// Interface for dolfinx/common
void common(nb::module_& m)
{
  // From dolfinx/common/defines.h
  m.attr("git_commit_hash") = dolfinx::git_commit_hash();
  m.attr("has_adios2") = dolfinx::has_adios2();
  m.attr("has_complex_ufcx_kernels") = dolfinx::has_complex_ufcx_kernels();
  m.attr("has_debug") = dolfinx::has_debug();
  m.attr("has_kahip") = dolfinx::has_kahip();
  m.attr("has_parmetis") = dolfinx::has_parmetis();
  m.attr("has_petsc") = dolfinx::has_petsc();
  m.attr("has_petsc4py") = has_petsc4py();
  m.attr("has_ptscotch") = dolfinx::has_ptscotch();
  m.attr("has_slepc") = dolfinx::has_slepc();
  m.attr("ufcx_signature") = dolfinx::ufcx_signature();
  m.attr("version") = dolfinx::version();

  nb::enum_<dolfinx::Table::Reduction>(m, "Reduction")
      .value("max", dolfinx::Table::Reduction::max)
      .value("min", dolfinx::Table::Reduction::min)
      .value("average", dolfinx::Table::Reduction::average);

  auto sc = nb::class_<dolfinx::common::Scatterer<>>(m, "Scatterer")
                .def(nb::init<dolfinx::common::IndexMap&, int>(),
                     nb::arg("index_map"), nb::arg("block_size"));
  add_scatter_functions<std::int64_t>(sc);
  add_scatter_functions<double>(sc);
  add_scatter_functions<float>(sc);

  // dolfinx::common::IndexMap
  nb::class_<dolfinx::common::IndexMap>(m, "IndexMap")
      .def(
          "__init__",
          [](dolfinx::common::IndexMap* self, MPICommWrapper comm,
             std::int32_t local_size)
          { new (self) dolfinx::common::IndexMap(comm.get(), local_size); },
          nb::arg("comm"), nb::arg("local_size"))
      .def(
          "__init__",
          [](dolfinx::common::IndexMap* self, MPICommWrapper comm,
             std::int32_t local_size,
             nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig> ghosts,
             nb::ndarray<const int, nb::ndim<1>, nb::c_contig> ghost_owners,
             int tag)
          {
            new (self) dolfinx::common::IndexMap(
                comm.get(), local_size, std::span(ghosts.data(), ghosts.size()),
                std::span(ghost_owners.data(), ghost_owners.size()), tag);
          },
          nb::arg("comm"), nb::arg("local_size"), nb::arg("ghosts"),
          nb::arg("ghost_owners"), nb::arg("tag"))
      .def(
          "__init__",
          [](dolfinx::common::IndexMap* self, MPICommWrapper comm,
             std::int32_t local_size,
             std::array<nb::ndarray<const int, nb::ndim<1>, nb::c_contig>, 2>
                 dest_src,
             nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig> ghosts,
             nb::ndarray<const int, nb::ndim<1>, nb::c_contig> ghost_owners)
          {
            std::array<std::vector<int>, 2> ranks;
            ranks[0].assign(dest_src[0].data(),
                            dest_src[0].data() + dest_src[0].size());
            ranks[1].assign(dest_src[1].data(),
                            dest_src[1].data() + dest_src[1].size());
            new (self) dolfinx::common::IndexMap(
                comm.get(), local_size, ranks,
                std::span(ghosts.data(), ghosts.size()),
                std::span(ghost_owners.data(), ghost_owners.size()));
          },
          nb::arg("comm"), nb::arg("local_size"), nb::arg("dest_src"),
          nb::arg("ghosts"), nb::arg("ghost_owners"))
      .def_prop_ro(
          "comm", [](const dolfinx::common::IndexMap& self)
          { return MPICommWrapper(self.comm()); }, nb::keep_alive<0, 1>())
      .def_prop_ro("size_local", &dolfinx::common::IndexMap::size_local)
      .def_prop_ro("size_global", &dolfinx::common::IndexMap::size_global)
      .def_prop_ro("num_ghosts", &dolfinx::common::IndexMap::num_ghosts)
      .def_prop_ro("local_range", &dolfinx::common::IndexMap::local_range,
                   "Range of indices owned by this map")
      .def("index_to_dest_ranks",
           &dolfinx::common::IndexMap::index_to_dest_ranks)
      .def_prop_ro(
          "ghosts",
          [](const dolfinx::common::IndexMap& self)
          {
            std::span ghosts = self.ghosts();
            return nb::ndarray<const std::int64_t, nb::numpy>(ghosts.data(),
                                                              {ghosts.size()});
          },
          nb::rv_policy::reference_internal, "Return list of ghost indices")
      .def_prop_ro(
          "owners",
          [](const dolfinx::common::IndexMap& self)
          {
            std::span owners = self.owners();
            return nb::ndarray<const int, nb::ndim<1>, nb::numpy>(
                owners.data(), {owners.size()});
          },
          nb::rv_policy::reference_internal)
      .def(
          "local_to_global",
          [](const dolfinx::common::IndexMap& self,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> local)
          {
            std::vector<std::int64_t> global(local.size());
            self.local_to_global(std::span(local.data(), local.size()), global);
            return dolfinx_wrappers::as_nbarray(std::move(global));
          },
          nb::arg("local"))
      .def(
          "global_to_local",
          [](const dolfinx::common::IndexMap& self,
             nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig> global)
          {
            std::vector<std::int32_t> local(global.size());
            self.global_to_local(std::span(global.data(), global.size()),
                                 local);
            return dolfinx_wrappers::as_nbarray(std::move(local));
          },
          nb::arg("global"))
      .def(
          "comm_graph", [](const dolfinx::common::IndexMap& self, int root)
          { return self.comm_graph(root); }, nb::arg("root") = 0,
          "Build a graph representing parallel communication patterns.")
      .def_static(
          "comm_graph_data",
          [](dolfinx::graph::AdjacencyList<
              std::tuple<int, std::size_t, std::int8_t>,
              std::pair<std::int32_t, std::int32_t>>& g)
          {
            std::vector<
                std::tuple<int, int, std::map<std::string, std::size_t>>>
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
                  return std::pair(n, std::map<std::string, std::int32_t>{
                                          {"weight", data.first},
                                          {"num_ghosts", data.second}});
                });

            return std::pair(std::move(adj), std::move(nodes));
          },
          nb::arg("root") = 0,
          "Build a graph representing parallel communication patterns.")
      .def_static(
          "comm_to_json",
          [](dolfinx::graph::AdjacencyList<
              std::tuple<int, std::size_t, std::int8_t>,
              std::pair<std::int32_t, std::int32_t>>& g)
          { return dolfinx::common::comm_to_json(g); },
          "Build a JSON string representation of a parallel communication "
          "graph that can use used by build a NetworkX graph.");

  // dolfinx::common::Timer
  nb::class_<dolfinx::common::Timer<std::chrono::high_resolution_clock>>(
      m, "Timer", "Timer class")
      .def(nb::init<std::optional<std::string>>(), nb::arg("task").none())
      .def("start",
           &dolfinx::common::Timer<std::chrono::high_resolution_clock>::start,
           "Start timer")
      .def("elapsed",
           &dolfinx::common::Timer<
               std::chrono::high_resolution_clock>::elapsed<>,
           "Elapsed time")
      .def("stop",
           &dolfinx::common::Timer<std::chrono::high_resolution_clock>::stop<>,
           "Stop timer")
      .def("resume",
           &dolfinx::common::Timer<std::chrono::high_resolution_clock>::resume,
           "Resume timer")
      .def("flush",
           &dolfinx::common::Timer<std::chrono::high_resolution_clock>::flush,
           "Flush timer");

  m.def("timing", &dolfinx::timing);
  m.def("timings", &dolfinx::timings);

  m.def(
      "list_timings",
      [](MPICommWrapper comm, dolfinx::Table::Reduction reduction)
      { dolfinx::list_timings(comm.get(), reduction); }, nb::arg("comm"),
      nb::arg("reduction"));

  m.def(
      "init_logging",
      [](std::vector<std::string> args)
      {
        std::vector<char*> argv(args.size() + 1, nullptr);
        for (std::size_t i = 0; i < args.size(); ++i)
          argv[i] = const_cast<char*>(args[i].data());
        dolfinx::init_logging(args.size(), argv.data());
      },
      nb::arg("args"));

  m.def(
      "create_sub_index_map",
      [](const dolfinx::common::IndexMap& imap,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> indices,
         bool allow_owner_change)
      {
        auto [map, submap_to_map] = dolfinx::common::create_sub_index_map(
            imap, std::span(indices.data(), indices.size()),
            dolfinx::common::IndexMapOrder::any, allow_owner_change);
        return std::pair(std::move(map), dolfinx_wrappers::as_nbarray(
                                             std::move(submap_to_map)));
      },
      nb::arg("index_map"), nb::arg("indices"), nb::arg("allow_owner_change"));
}
} // namespace dolfinx_wrappers
