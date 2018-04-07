// Copyright (C) 2017 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "casters.h"
#include <dolfin/common/Variable.h>
#include <dolfin/log/Table.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Mesh.h>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

namespace dolfin_wrappers
{
void log(py::module& m)
{

  // dolfin::LogLevel enums
  py::enum_<dolfin::LogLevel>(m, "LogLevel", py::arithmetic())
      .value("DEBUG", dolfin::LogLevel::DBG)
      .value("TRACE", dolfin::LogLevel::TRACE)
      .value("PROGRESS", dolfin::LogLevel::PROGRESS)
      .value("INFO", dolfin::LogLevel::INFO)
      .value("WARNING", dolfin::LogLevel::WARNING)
      .value("ERROR", dolfin::LogLevel::ERROR)
      .value("CRITICAL", dolfin::LogLevel::CRITICAL);

  // dolfin::Table
  py::class_<dolfin::Table, std::shared_ptr<dolfin::Table>,
             dolfin::common::Variable>(m, "Table")
      .def(py::init<std::string>())
      .def("str", &dolfin::Table::str);

  // dolfin/log free functions
  m.def("info",
        [](const dolfin::common::Variable& v) { dolfin::log::info(v); });
  m.def("info", [](const dolfin::common::Variable& v, bool verbose) {
    dolfin::log::info(v, verbose);
  });
  m.def("info", [](std::string s) { dolfin::log::info(s); });
  m.def("info", [](const dolfin::parameter::Parameters& p, bool verbose) {
    dolfin::log::info(p, verbose);
  });
  m.def("info", [](const dolfin::mesh::Mesh& mesh,
                   bool verbose) { dolfin::log::info(mesh, verbose); },
        py::arg("mesh"), py::arg("verbose") = false);
  m.def("set_log_level", &dolfin::log::set_log_level);
  m.def("get_log_level", &dolfin::log::get_log_level);
  m.def("log", [](dolfin::LogLevel level, std::string s) {
    dolfin::log::log(level, s);
  });
}
}
