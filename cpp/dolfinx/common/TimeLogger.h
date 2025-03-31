// Copyright (C) 2003-2016 Anders Logg, 2015 Jan Blechta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Table.h"
#include "timing.h"
#include <chrono>
#include <map>
#include <mpi.h>
#include <string>
#include <utility>

namespace dolfinx::common
{
/// @brief Time logger maintaining data collected by Timer, if registered.
///
/// @note This is a monotstate, i.e. the data members are static and thus
/// timings are aggregated into a single map.
class TimeLogger
{
public:
  /// @brief Singleton access.
  /// @return Unique time logger object.
  static TimeLogger& instance();

  /// Register timing (for later summary)
  void register_timing(std::string task,
                       std::chrono::duration<double, std::ratio<1>> wall);

  /// Return a summary of timings and tasks in a Table
  Table timing_table() const;

  /// List a summary of timings and tasks. Reduction type is
  /// printed.
  /// @param comm MPI Communicator
  /// @param reduction Reduction type (min, max or average)
  void list_timings(MPI_Comm comm, Table::Reduction reduction) const;

  /// @brief Return timing.
  /// @param[in] task The task name to retrieve the timing for
  /// @return Values (count, total wall time) for given task.
  std::pair<int, std::chrono::duration<double, std::ratio<1>>>
  timing(std::string task) const;

  /// @brief Logged elapsed times.
  /// @return Elapsed [task id: (count, total wall time)].
  std::map<std::string,
           std::pair<int, std::chrono::duration<double, std::ratio<1>>>>
  timings() const;

private:
  /// Constructor
  TimeLogger() = default;

  // This class is used as a singleton and thus should not allow copies.
  TimeLogger(const TimeLogger&) = delete;

  // This class is used as a singleton and thus should not allow copies.
  TimeLogger& operator=(const TimeLogger&) = delete;

  /// Destructor
  ~TimeLogger() = default;

  // List of timings for tasks, map from string to (num_timings,
  // total_wall_time)
  std::map<std::string,
           std::pair<int, std::chrono::duration<double, std::ratio<1>>>>
      _timings;
};
} // namespace dolfinx::common
