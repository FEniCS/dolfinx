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
#include <spdlog/sinks/basic_file_sink.h>

#include <string>

namespace nb = nanobind;

namespace dolfinx_wrappers
{
void log(nb::module_& m)
{
  // log level enums
  nb::enum_<spdlog::level::level_enum>(m, "LogLevel", nb::is_arithmetic())
      .value("OFF", spdlog::level::level_enum::off)
      .value("INFO", spdlog::level::level_enum::info)
      .value("WARNING", spdlog::level::level_enum::warn)
      .value("ERROR", spdlog::level::level_enum::err);

  m.def(
      "set_output_file",
      [](std::string filename)
      {
        try
        {
          auto logger = spdlog::basic_logger_mt("dolfinx", filename.c_str());
          spdlog::set_default_logger(logger);
        }
        catch (const spdlog::spdlog_ex& ex)
        {
          std::cout << "Log init failed: " << ex.what() << std::endl;
        }
      },
      nb::arg("filename"));

  m.def(
      "set_thread_name",
      [](std::string thread_name)
      {
        std::string fmt
            = "[%Y-%m-%d %H:%M:%S.%e] [" + thread_name + "] [%l] %v";
        spdlog::set_pattern(fmt);
      },
      nb::arg("thread_name"));

  m.def(
      "set_log_level",
      [](spdlog::level::level_enum level) { spdlog::set_level(level); },
      nb::arg("level"));
  m.def("get_log_level", []() { return spdlog::get_level(); });
  m.def(
      "log",
      [](spdlog::level::level_enum level, std::string s)
      {
        switch (level)
        {
        case (spdlog::level::level_enum::info):
          spdlog::info(s.c_str());
          break;
        case (spdlog::level::level_enum::warn):
          spdlog::warn(s.c_str());
          break;
        case (spdlog::level::level_enum::err):
          spdlog::error(s.c_str());
          break;
        default:
          throw std::runtime_error("Log level not supported");
          break;
        }
      },
      nb::arg("level"), nb::arg("s"));
}
} // namespace dolfinx_wrappers
