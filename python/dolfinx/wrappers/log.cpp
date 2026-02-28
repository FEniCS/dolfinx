// Copyright (C) 2017 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfinx/common/log.h>
#include <iostream>
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
      .value("TRACE", spdlog::level::level_enum::trace)
      .value("DEBUG", spdlog::level::level_enum::debug)
      .value("INFO", spdlog::level::level_enum::info)
      .value("WARNING", spdlog::level::level_enum::warn)
      .value("ERROR", spdlog::level::level_enum::err)
      .value("CRITICAL", spdlog::level::level_enum::critical)
      .value("OFF", spdlog::level::level_enum::off);

  m.def(
      "set_output_file",
      [](const std::string& filename)
      {
        try
        {
          auto logger = spdlog::basic_logger_mt("dolfinx", filename.c_str());
          spdlog::set_default_logger(logger);
        }
        catch (const spdlog::spdlog_ex& ex)
        {
          std::cout << "Log init failed: " << ex.what() << "\n";
        }
      },
      nb::arg("filename"));

  m.def(
      "set_thread_name",
      [](const std::string& thread_name)
      {
        std::string fmt
            = "[%Y-%m-%d %H:%M:%S.%e] [" + thread_name + "] [%l] %v";
        spdlog::set_pattern(fmt);
      },
      nb::arg("thread_name"));

  m.def(
      "set_log_level", [](spdlog::level::level_enum level)
      { spdlog::set_level(level); }, nb::arg("level"));
  m.def("get_log_level", []() { return spdlog::get_level(); });
  m.def(
      "log",
      [](spdlog::level::level_enum level, const std::string& s)
      {
        switch (level)
        {
        case (spdlog::level::level_enum::trace):
          spdlog::trace(s.c_str());
          break;
        case (spdlog::level::level_enum::debug):
          spdlog::debug(s.c_str());
          break;
        case (spdlog::level::level_enum::info):
          spdlog::info(s.c_str());
          break;
        case (spdlog::level::level_enum::warn):
          spdlog::warn(s.c_str());
          break;
        case (spdlog::level::level_enum::err):
          spdlog::error(s.c_str());
          break;
        case (spdlog::level::level_enum::critical):
          spdlog::critical(s.c_str());
          break;
        case (spdlog::level::level_enum::off):
          break;
        default:
          throw std::runtime_error("Log level not supported");
          break;
        }
      },
      nb::arg("level"), nb::arg("s"));
}
} // namespace dolfinx_wrappers
