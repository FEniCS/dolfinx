// Copyright (C) 2017 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <Eigen/Dense>
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/SubSystemsManager.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/Variable.h>
#include <dolfin/common/constants.h>
#include <dolfin/common/defines.h>
#include <dolfin/common/timing.h>
#include <dolfin/log/Table.h>
#include <memory>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <set>
#include <string>
#include <vector>

#include "MPICommWrapper.h"
#include "casters.h"

namespace py = pybind11;

namespace dolfin_wrappers
{
// Interface for dolfin/common
void common(py::module& m)
{
  // dolfin::common::Variable
  py::class_<dolfin::common::Variable,
             std::shared_ptr<dolfin::common::Variable>>(m, "Variable",
                                                        "Variable base class")
      .def("id", &dolfin::common::Variable::id)
      .def("name", &dolfin::common::Variable::name)
      .def("rename", &dolfin::common::Variable::rename)
      .def_readwrite("parameters", &dolfin::common::Variable::parameters);

  // From dolfin/common/defines.h
  m.def("has_debug", &dolfin::has_debug);
  m.def("has_hdf5", &dolfin::has_hdf5);
  m.def("has_hdf5_parallel", &dolfin::has_hdf5_parallel);
  m.def("has_mpi", &dolfin::has_mpi);
  m.def("has_mpi4py",
        []() {
#ifdef HAS_PYBIND11_MPI4PY
          return true;
#else
          return false;
#endif
        },
        "Return `True` if DOLFIN is configured with mpi4py");
  m.def("has_parmetis", &dolfin::has_parmetis);
  m.def("has_scotch", &dolfin::has_scotch);
  m.def("has_petsc_complex", &dolfin::has_petsc_complex,
        "Return `True` if DOLFIN is configured with PETSc compiled with "
        "scalars represented as complex numbers ");
  m.def("has_slepc", &dolfin::has_slepc,
        "Return `True` if DOLFIN is configured with SLEPc");
  m.def("has_petsc4py",
        []() {
#ifdef HAS_PYBIND11_PETSC4PY
          return true;
#else
          return false;
#endif
        },
        "Return `True` if DOLFIN is configured with petsc4py");
  m.def("has_slepc4py",
        []() {
#ifdef HAS_PYBIND11_SLEPC4PY
          return true;
#else
          return false;
#endif
        },
        "Return `True` if DOLFIN is configured with slepc4py");
  m.def("git_commit_hash", &dolfin::git_commit_hash,
        "Returns git hash for this build.");
  m.def("sizeof_la_index_t", &dolfin::sizeof_la_index_t);

  m.attr("DOLFIN_EPS") = DOLFIN_EPS;
  m.attr("DOLFIN_PI") = DOLFIN_PI;

  // dolfin::common::IndexMap
  py::class_<dolfin::common::IndexMap,
             std::shared_ptr<dolfin::common::IndexMap>>(m, "IndexMap")
      .def("size_local", &dolfin::common::IndexMap::size_local)
      .def("size_global", &dolfin::common::IndexMap::size_global)
      .def("num_ghosts", &dolfin::common::IndexMap::num_ghosts)
      .def("block_size", &dolfin::common::IndexMap::block_size,
           "Return block size")
      .def("local_range", &dolfin::common::IndexMap::local_range,
           "Range of indices owned by this map")
      .def("ghost_owners", &dolfin::common::IndexMap::ghost_owners,
           py::return_value_policy::reference_internal,
           "Return owning process for each ghost index")
      .def("ghosts", &dolfin::common::IndexMap::ghosts,
           py::return_value_policy::reference_internal,
           "Return list of ghost indices");

  // dolfin::common::Timer
  py::class_<dolfin::common::Timer, std::shared_ptr<dolfin::common::Timer>>(
      m, "Timer", "Timer class")
      .def(py::init<>())
      .def(py::init<std::string>())
      .def("start", &dolfin::common::Timer::start, "Start timer")
      .def("stop", &dolfin::common::Timer::stop, "Stop timer")
      .def("resume", &dolfin::common::Timer::resume)
      .def("elapsed", &dolfin::common::Timer::elapsed);

  // dolfin::common::Timer enum
  py::enum_<dolfin::TimingType>(m, "TimingType")
      .value("wall", dolfin::TimingType::wall)
      .value("system", dolfin::TimingType::system)
      .value("user", dolfin::TimingType::user);

  // dolfin/common free functions
  m.def("timing", &dolfin::timing);
  m.def("timings", [](std::vector<dolfin::TimingType> type) {
    std::set<dolfin::TimingType> _type(type.begin(), type.end());
    return dolfin::timings(_type);
  });
  m.def("list_timings", [](std::vector<dolfin::TimingType> type) {
    std::set<dolfin::TimingType> _type(type.begin(), type.end());
    dolfin::list_timings(_type);
  });

  // dolfin::SubSystemsManager
  py::class_<dolfin::common::SubSystemsManager,
             std::unique_ptr<dolfin::common::SubSystemsManager, py::nodelete>>(
      m, "SubSystemsManager")
      .def_static("init_petsc",
                  (void (*)()) & dolfin::common::SubSystemsManager::init_petsc)
      .def_static("init_petsc",
                  [](std::vector<std::string> args) {
                    std::vector<char*> argv(args.size());
                    for (std::size_t i = 0; i < args.size(); ++i)
                      argv[i] = const_cast<char*>(args[i].data());
                    dolfin::common::SubSystemsManager::init_petsc(args.size(),
                                                                  argv.data());
                  })
      .def_static("finalize", &dolfin::common::SubSystemsManager::finalize)
      .def_static("responsible_mpi",
                  &dolfin::common::SubSystemsManager::responsible_mpi)
      .def_static("responsible_petsc",
                  &dolfin::common::SubSystemsManager::responsible_petsc)
      .def_static("mpi_initialized",
                  &dolfin::common::SubSystemsManager::mpi_initialized)
      .def_static("mpi_finalized",
                  &dolfin::common::SubSystemsManager::mpi_finalized);
}

// Interface for MPI
void mpi(py::module& m)
{

#ifndef HAS_PYBIND11_MPI4PY
  // Expose the MPICommWrapper directly since we cannot cast it to
  // mpi4py
  py::class_<MPICommWrapper>(
      m, "MPICommWrapper",
      "DOLFIN is compiled without support for mpi4py. This object can be "
      "passed into DOLFIN as an MPI communicator, but is not an mpi4py comm.")
      .def("underlying_comm",
           [](MPICommWrapper self) { return (std::uintptr_t)self.get(); },
           "Return the underlying MPI_Comm cast to std::uintptr_t. "
           "The return value may or may not make sense depending on the MPI "
           "implementation.");
#endif

  // dolfin::MPI
  py::class_<dolfin::MPI>(m, "MPI", "MPI utilities")
      .def_property_readonly_static(
          "comm_world",
          [](py::object) { return MPICommWrapper(MPI_COMM_WORLD); })
      .def_property_readonly_static(
          "comm_self", [](py::object) { return MPICommWrapper(MPI_COMM_SELF); })
      .def_property_readonly_static(
          "comm_null", [](py::object) { return MPICommWrapper(MPI_COMM_NULL); })
      .def_static("init",
                  (void (*)()) & dolfin::common::SubSystemsManager::init_mpi,
                  "Initialise MPI")
      .def_static(
          "init",
          [](std::vector<std::string> args, int required_thread_level) -> int {
            std::vector<char*> argv(args.size());
            for (std::size_t i = 0; i < args.size(); ++i)
              argv[i] = const_cast<char*>(args[i].data());
            return dolfin::common::SubSystemsManager::init_mpi(
                args.size(), argv.data(), required_thread_level);
          },
          "Initialise MPI with command-line args and required level "
          "of thread support. Return provided thread level.")
      .def_static("responsible",
                  &dolfin::common::SubSystemsManager::responsible_mpi,
                  "Return true if DOLFIN initialised MPI (and is therefore "
                  "responsible for finalization)")
      .def_static("initialized",
                  &dolfin::common::SubSystemsManager::mpi_initialized,
                  "Check if MPI has been initialised")
      .def_static("finalized",
                  &dolfin::common::SubSystemsManager::mpi_finalized,
                  "Check if MPI has been finalized")
      .def_static("barrier",
                  [](const MPICommWrapper comm) {
                    return dolfin::MPI::barrier(comm.get());
                  })
      .def_static("rank",
                  [](const MPICommWrapper comm) {
                    return dolfin::MPI::rank(comm.get());
                  })
      .def_static("size",
                  [](const MPICommWrapper comm) {
                    return dolfin::MPI::size(comm.get());
                  })
      .def_static("local_range",
                  [](MPICommWrapper comm, std::int64_t N) {
                    return dolfin::MPI::local_range(comm.get(), N);
                  })
      // templated for double
      .def_static("max",
                  [](const MPICommWrapper comm, double value) {
                    return dolfin::MPI::max(comm.get(), value);
                  })
      .def_static("min",
                  [](const MPICommWrapper comm, double value) {
                    return dolfin::MPI::min(comm.get(), value);
                  })
      .def_static("sum",
                  [](const MPICommWrapper comm, double value) {
                    return dolfin::MPI::sum(comm.get(), value);
                  })
      // templated for dolfin::Table
      .def_static("max",
                  [](const MPICommWrapper comm, dolfin::Table value) {
                    return dolfin::MPI::max(comm.get(), value);
                  })
      .def_static("min",
                  [](const MPICommWrapper comm, dolfin::Table value) {
                    return dolfin::MPI::min(comm.get(), value);
                  })
      .def_static("sum",
                  [](const MPICommWrapper comm, dolfin::Table value) {
                    return dolfin::MPI::sum(comm.get(), value);
                  })
      .def_static("avg", [](const MPICommWrapper comm, dolfin::Table value) {
        return dolfin::MPI::avg(comm.get(), value);
      });
}
} // namespace dolfin_wrappers
