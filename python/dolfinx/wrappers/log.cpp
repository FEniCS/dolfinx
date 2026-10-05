// Copyright (C) 2017-2026 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfinx/common/log.h>
#include <format>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <stdexcept>
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
          spdlog::set_default_logger(
              spdlog::basic_logger_mt("dolfinx", filename));
        }
        catch (const spdlog::spdlog_ex& ex)
        {
          throw std::runtime_error(
              std::format("Log initialisation failed: {}", ex.what()));
        }
      },
      nb::arg("filename"));

  m.def(
      "set_thread_name",
      [](const std::string& thread_name)
      {
        spdlog::set_pattern(
            std::format("[%Y-%m-%d %H:%M:%S.%e] [{}] [%l] %v", thread_name));
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
          spdlog::trace(s);
          break;
        case (spdlog::level::level_enum::debug):
          spdlog::debug(s);
          break;
        case (spdlog::level::level_enum::info):
          spdlog::info(s);
          break;
        case (spdlog::level::level_enum::warn):
          spdlog::warn(s);
          break;
        case (spdlog::level::level_enum::err):
          spdlog::error(s);
          break;
        case (spdlog::level::level_enum::critical):
          spdlog::critical(s);
          break;
        case (spdlog::level::level_enum::off):
          break;
        default:
          throw std::invalid_argument("Log level not supported.");
        }
      },
      nb::arg("level"), nb::arg("s"));
}
} // namespace dolfinx_wrappers
