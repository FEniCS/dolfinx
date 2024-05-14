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

enum LogLevel
{
  INFO = google::GLOG_INFO,
  WARNING = google::GLOG_WARNING,
  ERROR = google::GLOG_ERROR
};

namespace dolfinx_wrappers
{

void log(nb::module_& m)
{
  // log level enums
  nb::enum_<LogLevel>(m, "LogLevel")
      .value("INFO", LogLevel::INFO)
      .value("WARNING", LogLevel::WARNING)
      .value("ERROR", LogLevel::ERROR);

  m.def(
      "set_output_file",
      [](std::string filename)
      { google::SetLogDestination(google::INFO, filename.c_str()); },
      nb::arg("filename"));

   // m.def(
   //     "set_thread_name",
   //     [](std::string thread_name)
   //     { google::InstallPrefixFormatter(&PrefixFormatter); },
   //     nb::arg("thread_name"));

   m.def(
       "set_log_level", [](LogLevel level) { FLAGS_minloglevel = level; },
       nb::arg("level"));
   m.def("get_log_level",
         []() { return static_cast<LogLevel>(FLAGS_minloglevel); });
   m.def(
       "log",
       [](LogLevel level, std::string s)
       {
         switch (level)
         {
         case (LogLevel::INFO):
           LOG(INFO) << s;
           break;
         case (LogLevel::WARNING):
           LOG(WARNING) << s;
           break;
         case (LogLevel::ERROR):
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
