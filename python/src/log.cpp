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
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <dolfin/common/Variable.h>
#include <dolfin/log/log.h>
#include <dolfin/log/Table.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/multistage/MultiStageScheme.h>
#include "casters.h"

namespace py = pybind11;

namespace dolfin_wrappers
{
  void log(py::module& m)
  {
    // dolfin::Table
    py::class_<dolfin::Table, std::shared_ptr<dolfin::Table>>(m, "Table")
      .def(py::init<std::string>())
      .def("str", &dolfin::Table::str);

    // dolfin/log free functions
    m.def("info", [](const dolfin::Variable& v){ dolfin::info(v); });
    m.def("info", [](const dolfin::Variable& v, bool verbose){ dolfin::info(v, verbose); });
    m.def("info", [](std::string s){ dolfin::info(s); });
    m.def("info", [](const dolfin::Parameters& p, bool verbose){ dolfin::info(p, verbose); });
    m.def("info", [](const dolfin::Mesh& mesh, bool verbose){ dolfin::info(mesh, verbose); },
          py::arg("mesh"), py::arg("verbose")=false);
    m.def("info", [](const dolfin::MultiStageScheme& mms, bool verbose){ dolfin::info(mms, verbose); },
          py::arg("scheme"), py::arg("verbose")=false);
    m.def("set_log_level", &dolfin::set_log_level);
    m.def("get_log_level", &dolfin::get_log_level);

    // dolfin::LogLevel enums
    py::enum_<dolfin::LogLevel>(m, "LogLevel")
      .value("DEBUG", dolfin::LogLevel::DBG)
      .value("TRACE", dolfin::LogLevel::TRACE)
      .value("PROGRESS", dolfin::LogLevel::PROGRESS)
      .value("INFO", dolfin::LogLevel::INFO)
      .value("WARNING", dolfin::LogLevel::WARNING)
      .value("ERROR", dolfin::LogLevel::ERROR)
      .value("CRITICAL", dolfin::LogLevel::CRITICAL);
  }
}
