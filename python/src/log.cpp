// Copyright (C) 2017 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "casters.h"
#include <dolfin/common/log.h>
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

  m.def("set_output_file", [](std::string filename) {
    loguru::add_file(filename.c_str(), loguru::Truncate,
                     loguru::Verbosity_INFO);
  });

  m.def("set_log_level", [](loguru::NamedVerbosity level) {
    loguru::g_stderr_verbosity = level;
  });
  m.def("get_log_level",
        []() { return loguru::NamedVerbosity(loguru::g_stderr_verbosity); });
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
    default:
      throw std::runtime_error("Log level not supported");
      break;
    }
  });
}
} // namespace dolfin_wrappers
