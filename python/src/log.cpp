// Copyright (C) 2017 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "casters.h"
#include <dolfin/common/Variable.h>
#include <dolfin/mesh/Mesh.h>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <spdlog/spdlog.h>
#include <string>

namespace py = pybind11;

namespace dolfin_wrappers
{
void log(py::module& m)
{

  // log level enums
  py::enum_<spdlog::level::level_enum>(m, "LogLevel", py::arithmetic())
      .value("TRACE", spdlog::level::trace)
      .value("DEBUG", spdlog::level::debug)
      .value("INFO", spdlog::level::info)
      .value("WARNING", spdlog::level::warn)
      .value("ERROR", spdlog::level::err)
      .value("CRITICAL", spdlog::level::critical);

  // dolfin/log free functions
  m.def("info",
        [](const dolfin::common::Variable& v) { spdlog::info(v.str(false)); });
  m.def("info", [](const dolfin::common::Variable& v, bool verbose) {
    spdlog::info(v.str(verbose));
  });
  m.def("info", [](std::string s) { spdlog::info(s); });
  m.def("info",
        [](const dolfin::mesh::Mesh& mesh, bool verbose) {
          spdlog::info(mesh.str(verbose));
        },
        py::arg("mesh"), py::arg("verbose") = false);
  m.def("set_log_level", &spdlog::set_level);
  m.def("get_log_level", []() { return spdlog::default_logger()->level(); });
  m.def("log", [](spdlog::level::level_enum level, std::string s) {
    // FIXME: there must be a better way to do this...
    switch (level)
    {
    case spdlog::level::trace:
      spdlog::trace(s);
      break;
    case spdlog::level::debug:
      spdlog::debug(s);
      break;
    case spdlog::level::info:
      spdlog::info(s);
      break;
    case spdlog::level::warn:
      spdlog::warn(s);
      break;
    case spdlog::level::err:
      spdlog::error(s);
      break;
    case spdlog::level::critical:
      spdlog::critical(s);
      break;
    default:
      spdlog::warn(s);
      break;
    }
  });
}
} // namespace dolfin_wrappers
