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

#include <glog/log_severity.h>

namespace nb = nanobind;

enum LogLevel
{
  INFO = 0,
  WARNING = 1,
  ERROR = 2
};

namespace dolfinx_wrappers
{
void log(nb::module_& m)
{
  // log level enums
  nb::enum_<LogLevel>(m, "LogLevel")
      .value("INFO", google::GLOG_INFO)
      .value("WARNING", google::GLOG_WARNING)
      .value("ERROR", google::GLOG_ERROR);

  m.def(
      "set_output_file", [](std::string filename)
      { google::SetLogDestination(google::INFO, filename.c_str()); },
      nb::arg("filename"));

  // m.def(
  //     "set_thread_name", [](std::string thread_name)
  //     { google::LogSeverity::set_thread_name(thread_name.c_str()); },
  //     nb::arg("thread_name"));

  m.def(
      "set_log_level", [](int level) { FLAGS_minloglevel = level; },
      nb::arg("level"));
  m.def("get_log_level", []() { return FLAGS_minloglevel; });
  m.def(
      "log",
      [](int level, std::string s)
      {
        switch (level)
        {
        case (google::GLOG_INFO):
          LOG(INFO) << s;
          break;
        case (google::GLOG_WARNING):
          LOG(WARNING) << s;
          break;
        case (google::GLOG_ERROR):
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
