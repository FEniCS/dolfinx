// Copyright (C) 2017-2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MPICommWrapper.h"
#include "array.h"
#include "caster_mpi.h"
#include "caster_petsc.h"
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
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <span>
#include <string>
#include <vector>

namespace py = pybind11;

namespace dolfinx_wrappers
{
// Interface for dolfinx/common
void common(py::module& m)
{
  // From dolfinx/common/defines.h
  m.attr("has_debug") = dolfinx::has_debug();
  m.attr("has_parmetis") = dolfinx::has_parmetis();
  m.attr("has_kahip") = dolfinx::has_kahip();
  m.attr("has_slepc") = dolfinx::has_slepc();
  m.attr("has_adios2") = dolfinx::has_adios2();

#ifdef HAS_PYBIND11_SLEPC4PY
  m.attr("has_slepc4py") = true;
#else
  m.attr("has_slepc4py") = false;
#endif
  m.attr("git_commit_hash") = dolfinx::git_commit_hash();

  py::enum_<dolfinx::Table::Reduction>(m, "Reduction")
      .value("max", dolfinx::Table::Reduction::max)
      .value("min", dolfinx::Table::Reduction::min)
      .value("average", dolfinx::Table::Reduction::average);

  // dolfinx::common::IndexMap
  py::class_<dolfinx::common::IndexMap,
             std::shared_ptr<dolfinx::common::IndexMap>>(m, "IndexMap")
      .def(py::init(
               [](const MPICommWrapper comm, std::int32_t local_size)
               { return dolfinx::common::IndexMap(comm.get(), local_size); }),
           py::arg("comm"), py::arg("local_size"))
      .def(py::init(
               [](const MPICommWrapper comm, std::int32_t local_size,
                  const py::array_t<std::int64_t, py::array::c_style>& ghosts,
                  const py::array_t<int, py::array::c_style>& ghost_owners)
               {
                 return dolfinx::common::IndexMap(
                     comm.get(), local_size,
                     std::span(ghosts.data(), ghosts.size()),
                     std::span(ghost_owners.data(), ghost_owners.size()));
               }),
           py::arg("comm"), py::arg("local_size"), py::arg("ghosts"),
           py::arg("ghost_owners"))
      .def(py::init(
               [](const MPICommWrapper comm, std::int32_t local_size,
                  const std::array<py::array_t<int, py::array::c_style>, 2>&
                      dest_src,
                  const py::array_t<std::int64_t, py::array::c_style>& ghosts,
                  const py::array_t<int, py::array::c_style>& ghost_owners)
               {
                 std::array<std::vector<int>, 2> ranks;
                 ranks[0].assign(dest_src[0].data(),
                                 dest_src[0].data() + dest_src[0].size());
                 ranks[1].assign(dest_src[1].data(),
                                 dest_src[1].data() + dest_src[1].size());
                 return dolfinx::common::IndexMap(
                     comm.get(), local_size, ranks,
                     std::span(ghosts.data(), ghosts.size()),
                     std::span(ghost_owners.data(), ghost_owners.size()));
               }),
           py::arg("comm"), py::arg("local_size"), py::arg("dest_src"),
           py::arg("ghosts"), py::arg("ghost_owners"))
      .def_property_readonly("size_local",
                             &dolfinx::common::IndexMap::size_local)
      .def_property_readonly("size_global",
                             &dolfinx::common::IndexMap::size_global)
      .def_property_readonly("num_ghosts",
                             &dolfinx::common::IndexMap::num_ghosts)
      .def_property_readonly("local_range",
                             &dolfinx::common::IndexMap::local_range,
                             "Range of indices owned by this map")
      .def_property_readonly(
          "ghosts",
          [](const dolfinx::common::IndexMap& self)
          {
            const std::vector<std::int64_t>& ghosts = self.ghosts();
            return py::array_t<std::int64_t>(ghosts.size(), ghosts.data(),
                                             py::cast(self));
          },
          "Return list of ghost indices")
      .def_property_readonly("owners",
                             [](const dolfinx::common::IndexMap& self)
                             {
                               const std::vector<int>& owners = self.owners();
                               return py::array_t<int>(owners.size(),
                                                       owners.data(),
                                                       py::cast(self));
                             })
      .def(
          "local_to_global",
          [](const dolfinx::common::IndexMap& self,
             const py::array_t<std::int32_t, py::array::c_style>& local)
          {
            if (local.ndim() != 1)
              throw std::runtime_error("Array of local indices must be 1D.");
            py::array_t<std::int64_t> global(local.size());
            self.local_to_global(
                std::span(local.data(), local.size()),
                std::span<std::int64_t>(global.mutable_data(), global.size()));
            return global;
          },
          py::arg("local"))
      .def(
          "create_submap",
          [](const dolfinx::common::IndexMap& self,
             const py::array_t<std::int32_t, py::array::c_style>& entities)
          {
            auto [map, ghosts] = self.create_submap(
                std::span(entities.data(), entities.size()));
            return std::pair(std::move(map), as_pyarray(std::move(ghosts)));
          },
          py::arg("entities"));

  // dolfinx::common::Timer
  py::class_<dolfinx::common::Timer, std::shared_ptr<dolfinx::common::Timer>>(
      m, "Timer", "Timer class")
      .def(py::init<>())
      .def(py::init<std::string>(), py::arg("task"))
      .def("start", &dolfinx::common::Timer::start, "Start timer")
      .def("stop", &dolfinx::common::Timer::stop, "Stop timer")
      .def("resume", &dolfinx::common::Timer::resume)
      .def("elapsed", &dolfinx::common::Timer::elapsed);

  // dolfinx::common::Timer enum
  py::enum_<dolfinx::TimingType>(m, "TimingType")
      .value("wall", dolfinx::TimingType::wall)
      .value("system", dolfinx::TimingType::system)
      .value("user", dolfinx::TimingType::user);

  m.def("timing", &dolfinx::timing);

  m.def(
      "list_timings",
      [](const MPICommWrapper comm, std::vector<dolfinx::TimingType> type,
         dolfinx::Table::Reduction reduction)
      {
        std::set<dolfinx::TimingType> _type(type.begin(), type.end());
        dolfinx::list_timings(comm.get(), _type, reduction);
      },
      py::arg("comm"), py::arg("type"), py::arg("reduction"));

  m.def(
      "init_logging",
      [](std::vector<std::string> args)
      {
        std::vector<char*> argv(args.size() + 1, nullptr);
        for (std::size_t i = 0; i < args.size(); ++i)
          argv[i] = const_cast<char*>(args[i].data());
        dolfinx::init_logging(args.size(), argv.data());
      },
      py::arg("args"));
}
} // namespace dolfinx_wrappers
