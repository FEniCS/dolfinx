// Copyright (C) 2017-2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MPICommWrapper.h"
#include "caster_mpi.h"
#include "caster_petsc.h"
#include <complex>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/Table.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/defines.h>
#include <dolfinx/common/subsystem.h>
#include <dolfinx/common/timing.h>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

namespace py = pybind11;

namespace dolfinx_wrappers
{
// Interface for dolfin/common
void common(py::module& m)
{
  // From dolfin/common/defines.h
  m.attr("has_debug") = dolfinx::has_debug();
  m.attr("has_parmetis") = dolfinx::has_parmetis();
  m.attr("has_kahip") = dolfinx::has_kahip();
  m.attr("has_petsc_complex") = dolfinx::has_petsc_complex();
  m.attr("has_slepc") = dolfinx::has_slepc();
#ifdef HAS_PYBIND11_SLEPC4PY
  m.attr("has_slepc4py") = true;
#else
  m.attr("has_slepc4py") = false;
#endif
  m.attr("git_commit_hash") = dolfinx::git_commit_hash();

  // dolfinx::common::IndexMap::Mode
  py::enum_<dolfinx::common::IndexMap::Mode>(m, "ScatterMode")
      .value("add", dolfinx::common::IndexMap::Mode::add)
      .value("insert", dolfinx::common::IndexMap::Mode::insert);

  // dolfinx::common::IndexMap
  py::class_<dolfinx::common::IndexMap,
             std::shared_ptr<dolfinx::common::IndexMap>>(m, "IndexMap")
      .def(py::init([](const MPICommWrapper comm, std::int32_t local_size,
                       const std::vector<int>& dest_ranks,
                       const std::vector<std::int64_t>& ghosts,
                       const std::vector<int>& ghost_owners) {
        return std::make_shared<dolfinx::common::IndexMap>(
            comm.get(), local_size, dest_ranks, ghosts, ghost_owners);
      }))
      .def_property_readonly("size_local",
                             &dolfinx::common::IndexMap::size_local)
      .def_property_readonly("size_global",
                             &dolfinx::common::IndexMap::size_global)
      .def_property_readonly("num_ghosts",
                             &dolfinx::common::IndexMap::num_ghosts)
      .def_property_readonly("local_range",
                             &dolfinx::common::IndexMap::local_range,
                             "Range of indices owned by this map")
      .def("ghost_owner_rank", &dolfinx::common::IndexMap::ghost_owner_rank,
           "Return owning process for each ghost index")
      .def_property_readonly(
          "ghosts",
          [](const dolfinx::common::IndexMap& self) {
            const std::vector<std::int64_t>& ghosts = self.ghosts();
            return py::array_t<std::int64_t>(ghosts.size(), ghosts.data(),
                                             py::cast(self));
          },
          "Return list of ghost indices")
      .def("global_indices", &dolfinx::common::IndexMap::global_indices)
      .def("local_to_global",
           [](const dolfinx::common::IndexMap& self,
              const py::array_t<std::int32_t, py::array::c_style>& local) {
             if (local.ndim() != 1)
               throw std::runtime_error("Array of local indices must be 1D.");
             py::array_t<std::int64_t> global(local.size());
             self.local_to_global(local.data(), local.size(),
                                  global.mutable_data());
             return global;
           });

  // dolfinx::common::Timer
  py::class_<dolfinx::common::Timer, std::shared_ptr<dolfinx::common::Timer>>(
      m, "Timer", "Timer class")
      .def(py::init<>())
      .def(py::init<std::string>())
      .def("start", &dolfinx::common::Timer::start, "Start timer")
      .def("stop", &dolfinx::common::Timer::stop, "Stop timer")
      .def("resume", &dolfinx::common::Timer::resume)
      .def("elapsed", &dolfinx::common::Timer::elapsed);

  // dolfinx::common::Timer enum
  py::enum_<dolfinx::TimingType>(m, "TimingType")
      .value("wall", dolfinx::TimingType::wall)
      .value("system", dolfinx::TimingType::system)
      .value("user", dolfinx::TimingType::user);

  // dolfin/common free functions
  m.def("timing", &dolfinx::timing);

  m.def("list_timings",
        [](const MPICommWrapper comm, std::vector<dolfinx::TimingType> type) {
          std::set<dolfinx::TimingType> _type(type.begin(), type.end());
          dolfinx::list_timings(comm.get(), _type);
        });

  m.def("init_logging", [](std::vector<std::string> args) {
    std::vector<char*> argv(args.size() + 1, nullptr);
    for (std::size_t i = 0; i < args.size(); ++i)
      argv[i] = const_cast<char*>(args[i].data());
    dolfinx::common::subsystem::init_logging(args.size(), argv.data());
  });
}
} // namespace dolfinx_wrappers
