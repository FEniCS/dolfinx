// Copyright (C) 2017 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

#include <memory>
#include <set>
#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <dolfin/common/MPI.h>
#include <dolfin/common/constants.h>
#include <dolfin/common/defines.h>
#include <dolfin/common/SubSystemsManager.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/Variable.h>
#include <dolfin/common/timing.h>
#include <dolfin/log/Table.h>

#include "casters.h"

namespace py = pybind11;

namespace dolfin_wrappers
{
  // Interface for dolfin/common
  void common(py::module& m)
  {
    // dolfin::Variable
    py::class_<dolfin::Variable, std::shared_ptr<dolfin::Variable>>
      (m, "Variable", "Variable base class")
      .def("id", &dolfin::Variable::id)
      .def("name", &dolfin::Variable::name)
      .def("label", &dolfin::Variable::label)
      .def("rename", &dolfin::Variable::rename)
      .def_readwrite("parameters", &dolfin::Variable::parameters);

    // From dolfin/common/defines.h
    m.def("has_debug", &dolfin::has_debug);
    m.def("has_hdf5", &dolfin::has_hdf5);
    m.def("has_hdf5_parallel", &dolfin::has_hdf5_parallel);
    m.def("has_mpi", &dolfin::has_mpi);
    m.def("has_parmetis", &dolfin::has_parmetis);
    m.def("has_scotch", &dolfin::has_scotch);
    m.def("has_petsc", &dolfin::has_petsc, "Return `True` if DOLFIN is configured with PETSc");
    m.def("has_slepc", &dolfin::has_slepc, "Return `True` if DOLFIN is configured with SLEPc");
    m.def("has_petsc4py", []()
          {
            #ifdef HAS_PYBIND11_PETSC4PY
            return true;
            #else
            return false;
            #endif
          }, "Return `True` if DOLFIN is configured with petsc4py");
    m.def("has_slepc4py", []()
          {
            #ifdef HAS_PYBIND11_SLEPC4PY
            return true;
            #else
            return false;
            #endif
          }, "Return `True` if DOLFIN is configured with slepc4py");
    m.def("git_commit_hash", &dolfin::git_commit_hash, "Returns git hash for this build.");
    m.def("sizeof_la_index", &dolfin::sizeof_la_index);

    m.attr("DOLFIN_EPS") = DOLFIN_EPS;
    m.attr("DOLFIN_PI") = DOLFIN_PI;

    // dolfin::Timer
    py::class_<dolfin::Timer, std::shared_ptr<dolfin::Timer>>
      (m, "Timer", "Timer class")
      .def(py::init<>())
      .def(py::init<std::string>())
      .def("start", &dolfin::Timer::start, "Start timer")
      .def("stop", &dolfin::Timer::stop, "Stop timer")
      .def("resume", &dolfin::Timer::resume)
      .def("elapsed", &dolfin::Timer::elapsed);

    // dolfin::Timer enums
    py::enum_<dolfin::TimingClear>(m, "TimingClear")
      .value("clear", dolfin::TimingClear::clear)
      .value("keep", dolfin::TimingClear::keep);
    py::enum_<dolfin::TimingType>(m, "TimingType")
      .value("wall", dolfin::TimingType::wall)
      .value("system", dolfin::TimingType::system)
      .value("user", dolfin::TimingType::user);

    // dolfin/common free functions
    m.def("timing", &dolfin::timing);
    m.def("timings", [](dolfin::TimingClear clear, std::vector<dolfin::TimingType> type)
          {
            std::set<dolfin::TimingType> _type(type.begin(), type.end());
            return dolfin::timings(clear, _type);
          });
    m.def("list_timings", [](dolfin::TimingClear clear, std::vector<dolfin::TimingType> type)
          {
            std::set<dolfin::TimingType> _type(type.begin(), type.end());
            dolfin::list_timings(clear, _type);
          });
    m.def("dump_timings_to_xml", &dolfin::dump_timings_to_xml);

  }

  // Interface for MPI
  void mpi(py::module& m)
  {

    #ifndef HAS_MPI4PY
    py::class_<dolfin_wrappers::MPICommWrapper>(m, "MPICommWrapper"
      "DOLFIN is compiled without support for mpi4py. This object can be "
      "passed into DOLFIN as an MPI communicator, but is not an mpi4py comm.")
      .def("underlying_comm", [](dolfin_wrappers::MPICommWrapper &self)
        { return (std::uintptr_t) self.get(); },
        "Return the underlying MPI_Comm cast to std::uintptr_t. "
        "The return value may or may not make sense depending on the MPI implementation.");
    #endif

    // dolfin::MPI
    py::class_<dolfin::MPI>(m, "MPI", "MPI utilities")
      .def_property_readonly_static("comm_world", [](py::object)
        { return MPICommWrapper(MPI_COMM_WORLD); })
      .def_property_readonly_static("comm_self", [](py::object)
        { return MPICommWrapper(MPI_COMM_SELF); })
      .def_property_readonly_static("comm_null", [](py::object)
        { return MPICommWrapper(MPI_COMM_NULL); })
      .def_static("init", [](){ dolfin::SubSystemsManager::init_mpi(); }, "Initialise MPI")
      .def_static("barrier", [](const MPICommWrapper &comm)
        { return dolfin::MPI::barrier(comm.get()); })
      .def_static("rank", [](const MPICommWrapper &comm)
        { return dolfin::MPI::rank(comm.get()); })
      .def_static("size", [](const MPICommWrapper &comm)
        { return dolfin::MPI::size(comm.get()); })
      .def_static("local_range", [](MPICommWrapper comm, std::int64_t N)
        { return dolfin::MPI::local_range(comm.get(), N); })
      // templated for double
      .def_static("max", [](const MPICommWrapper &comm, double value)
        { return dolfin::MPI::max(comm.get(), value); })
      .def_static("min", [](const MPICommWrapper &comm, double value)
        { return dolfin::MPI::min(comm.get(), value); })
      .def_static("sum", [](const MPICommWrapper &comm, double value)
        { return dolfin::MPI::sum(comm.get(), value); })
      // templated for dolfin::Table
      .def_static("max", [](const MPICommWrapper &comm, dolfin::Table value)
        { return dolfin::MPI::max(comm.get(), value); })
      .def_static("min", [](const MPICommWrapper &comm, dolfin::Table value)
        { return dolfin::MPI::min(comm.get(), value); })
      .def_static("sum", [](const MPICommWrapper &comm, dolfin::Table value)
        { return dolfin::MPI::sum(comm.get(), value); })
      .def_static("avg", [](const MPICommWrapper &comm, dolfin::Table value)
        { return dolfin::MPI::avg(comm.get(), value); });
     }

}
