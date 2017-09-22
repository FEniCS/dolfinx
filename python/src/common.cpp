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
    m.def("has_petsc", &dolfin::has_petsc);
    m.def("has_slepc", &dolfin::has_slepc, "Return `True` if DOLFIN is configured with SLEPc");
    m.def("git_commit_hash", &dolfin::git_commit_hash, "Get git hash for this build.");
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
    /*
    #ifdef HAS_MPI4PY
    dolfin::SubSystemsManager::init_mpi();
    import_mpi4py();
    #endif
    */

    // dolfin::MPI
    py::class_<dolfin::MPI>(m, "MPI", "MPI utilities")
#ifdef OPEN_MPI
      .def_property_readonly_static("comm_world", [](py::object)
                                    { return reinterpret_cast<std::uintptr_t>(MPI_COMM_WORLD); })
      .def_property_readonly_static("comm_self", [](py::object)
                                    { return reinterpret_cast<std::uintptr_t>(MPI_COMM_SELF); })
      .def_property_readonly_static("comm_null", [](py::object)
                                    { return reinterpret_cast<std::uintptr_t>(MPI_COMM_NULL); })
#else
      .def_property_readonly_static("comm_world", [](py::object) { return MPI_COMM_WORLD; })
      .def_property_readonly_static("comm_self", [](py::object) { return MPI_COMM_SELF; })
      .def_property_readonly_static("comm_null", [](py::object) { return MPI_COMM_NULL; })
#endif
      .def_static("init", [](){ dolfin::SubSystemsManager::init_mpi(); }, "Initialise MPI")
      .def_static("barrier", &dolfin::MPI::barrier)
      .def_static("rank", &dolfin::MPI::rank)
      .def_static("size", &dolfin::MPI::size)
      .def_static("local_range", (std::pair<std::int64_t, std::int64_t> (*)(MPI_Comm, std::int64_t))
                  &dolfin::MPI::local_range)
      .def_static("max", &dolfin::MPI::max<double>)
      .def_static("min", &dolfin::MPI::min<double>)
      .def_static("sum", &dolfin::MPI::sum<double>)
      .def_static("min", &dolfin::MPI::min<dolfin::Table>)
      .def_static("max", &dolfin::MPI::max<dolfin::Table>)
      .def_static("sum", &dolfin::MPI::sum<dolfin::Table>)
      .def_static("avg", &dolfin::MPI::avg<dolfin::Table>)
      /*
#ifdef HAS_MPI4PY
      .def("to_mpi4py_comm", [](py::object obj){
          // If object is already a mpi4py communicator, return
          if (PyObject_TypeCheck(obj.ptr(), &PyMPIComm_Type))
            return obj;

          MPI_Comm comm_new;
          #ifdef OPEN_MPI
          std::uintptr_t c = obj.cast<std::uintptr_t>();
          MPI_Comm_dup(reinterpret_cast<MPI_Comm>(c), &comm_new);
          #else
          auto value = PyLong_AsLong(obj.ptr());
          MPI_Comm_dup(value, &comm_new);
          #endif

          // Create wrapper for conversion to mpi4py
          dolfin_wrappers::mpi_communicator mpi_comm;
          mpi_comm.comm = comm_new;

          return py::cast(mpi_comm);
        },
        "Convert a plain MPI communicator into a mpi4py communicator")
#endif
      */
      ;
     }

}
