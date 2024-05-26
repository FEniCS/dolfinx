// Copyright (C) 2017-2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MPICommWrapper.h"
#include "array.h"
#include "caster_mpi.h"
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
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <span>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace dolfinx_wrappers
{
// Interface for dolfinx/common
void common(nb::module_& m)
{
  // From dolfinx/common/defines.h
  m.attr("has_debug") = dolfinx::has_debug();
  m.attr("has_parmetis") = dolfinx::has_parmetis();
  m.attr("has_kahip") = dolfinx::has_kahip();
  m.attr("has_petsc") = dolfinx::has_petsc();
  m.attr("has_slepc") = dolfinx::has_slepc();
  m.attr("has_adios2") = dolfinx::has_adios2();
  m.attr("git_commit_hash") = dolfinx::git_commit_hash();

  nb::enum_<dolfinx::Table::Reduction>(m, "Reduction")
      .value("max", dolfinx::Table::Reduction::max)
      .value("min", dolfinx::Table::Reduction::min)
      .value("average", dolfinx::Table::Reduction::average);

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
             nb::ndarray<const int, nb::ndim<1>, nb::c_contig> ghost_owners)
          {
            new (self) dolfinx::common::IndexMap(
                comm.get(), local_size, std::span(ghosts.data(), ghosts.size()),
                std::span(ghost_owners.data(), ghost_owners.size()));
          },
          nb::arg("comm"), nb::arg("local_size"), nb::arg("ghosts"),
          nb::arg("ghost_owners"))
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
      .def("imbalance", &dolfinx::common::IndexMap::imbalance,
           "Imbalance of the current IndexMap.")
      .def_prop_ro(
          "ghosts",
          [](const dolfinx::common::IndexMap& self)
          {
            std::span ghosts = self.ghosts();
            return nb::ndarray<const std::int64_t, nb::numpy>(
                ghosts.data(), {ghosts.size()}, nb::handle());
          },
          nb::rv_policy::reference_internal, "Return list of ghost indices")
      .def_prop_ro(
          "owners",
          [](const dolfinx::common::IndexMap& self)
          {
            std::span owners = self.owners();
            return nb::ndarray<nb::numpy, const int, nb::ndim<1>>(
                owners.data(), {owners.size()}, nb::handle());
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
          nb::arg("global"));
  // dolfinx::common::Timer
  nb::class_<dolfinx::common::Timer>(m, "Timer", "Timer class")
      .def(nb::init<>())
      .def(nb::init<std::string>(), nb::arg("task"))
      .def("start", &dolfinx::common::Timer::start, "Start timer")
      .def("stop", &dolfinx::common::Timer::stop, "Stop timer")
      .def("resume", &dolfinx::common::Timer::resume)
      .def("elapsed", &dolfinx::common::Timer::elapsed);

  // dolfinx::common::Timer enum
  nb::enum_<dolfinx::TimingType>(m, "TimingType")
      .value("wall", dolfinx::TimingType::wall)
      .value("system", dolfinx::TimingType::system)
      .value("user", dolfinx::TimingType::user);

  m.def("timing", &dolfinx::timing);

  m.def(
      "list_timings",
      [](MPICommWrapper comm, std::vector<dolfinx::TimingType> type,
         dolfinx::Table::Reduction reduction)
      {
        std::set<dolfinx::TimingType> _type(type.begin(), type.end());
        dolfinx::list_timings(comm.get(), _type, reduction);
      },
      nb::arg("comm"), nb::arg("type"), nb::arg("reduction"));

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
