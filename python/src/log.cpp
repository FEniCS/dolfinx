// Copyright (C) 2017 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "casters.h"
#include <dolfin/common/Variable.h>
#include <dolfin/common/loguru.hpp>
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
  // log level enums
  py::enum_<loguru::NamedVerbosity>(m, "LogLevel", py::arithmetic())
      .value("OFF", loguru::Verbosity_OFF)
      .value("INFO", loguru::Verbosity_INFO)
      .value("WARNING", loguru::Verbosity_WARNING)
      .value("ERROR", loguru::Verbosity_ERROR);

  // dolfin/log free functions
  //  m.def("info",
  //       [](const dolfin::common::Variable& v) {
  //       spdlog::info(v.str(false));
  //       });
  // m.def("info", [](const dolfin::common::Variable& v, bool verbose) {
  //   spdlog::info(v.str(verbose));
  // });
  m.def("info", [](std::string s) { LOG(INFO) << s; });
  // m.def("info",
  //       [](const dolfin::mesh::Mesh& mesh, bool verbose) {
  //         spdlog::info(mesh.str(verbose));
  //       },
  //       py::arg("mesh"), py::arg("verbose") = false);
  m.def("set_log_level", [](loguru::NamedVerbosity level) {
    loguru::g_stderr_verbosity = level;
  });
  m.def("get_log_level", []() { return loguru::g_stderr_verbosity; });
  m.def("log", [](loguru::NamedVerbosity level, std::string s) {
    switch (level)
    {
    case (loguru::Verbosity_INFO):
      LOG(INFO) << s;
      break;
    case (loguru::Verbosity_WARNING):
      LOG(WARNING) << s;
      break;
    case (loguru::Verbosity_ERROR):
      LOG(ERROR) << s;
      break;
    }
  });
}
} // namespace dolfin_wrappers
