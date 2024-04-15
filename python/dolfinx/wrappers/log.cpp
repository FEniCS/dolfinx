// Copyright (C) 2017 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfinx/common/log.h>
#include <dolfinx/mesh/Mesh.h>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <string>

namespace nb = nanobind;

namespace dolfinx_wrappers
{
void log(nb::module_& m)
{
  // log level enums
  nb::enum_<loguru::NamedVerbosity>(m, "LogLevel", nb::is_arithmetic())
      .value("OFF", loguru::Verbosity_OFF)
      .value("INFO", loguru::Verbosity_INFO)
      .value("WARNING", loguru::Verbosity_WARNING)
      .value("ERROR", loguru::Verbosity_ERROR);

  m.def(
      "set_output_file",
      [](std::string filename)
      {
        loguru::add_file(filename.c_str(), loguru::Truncate,
                         loguru::Verbosity_INFO);
      },
      nb::arg("filename"));

  m.def(
      "set_thread_name", [](std::string thread_name)
      { loguru::set_thread_name(thread_name.c_str()); },
      nb::arg("thread_name"));

  m.def(
      "set_log_level", [](loguru::NamedVerbosity level)
      { loguru::g_stderr_verbosity = level; }, nb::arg("level"));
  m.def("get_log_level",
        []() { return loguru::NamedVerbosity(loguru::g_stderr_verbosity); });
  m.def(
      "log",
      [](loguru::NamedVerbosity level, std::string s)
      {
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
      },
      nb::arg("level"), nb::arg("s"));
}
} // namespace dolfinx_wrappers
