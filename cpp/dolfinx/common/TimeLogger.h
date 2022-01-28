// Copyright (C) 2003-2016 Anders Logg, 2015 Jan Blechta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/Table.h>
#include <dolfinx/common/timing.h>
#include <map>
#include <mpi.h>
#include <set>
#include <string>
#include <tuple>

namespace dolfinx::common
{

/// Timer logging

class TimeLogger
{
public:
  /// Constructor
  TimeLogger() = default;

  // This class is used as a singleton and thus should not allow copies.
  TimeLogger(const TimeLogger&) = delete;

  // This class is used as a singleton and thus should not allow copies.
  TimeLogger& operator=(const TimeLogger&) = delete;

  /// Destructor
  ~TimeLogger() = default;

  /// Register timing (for later summary)
  void register_timing(std::string task, double wall, double user,
                       double system);

  /// Return a summary of timings and tasks in a Table
  Table timings(std::set<TimingType> type);

  /// List a summary of timings and tasks. Reduction type is
  /// printed.
  /// @param comm MPI Communicator
  /// @param type Set of possible timings: wall, user or system
  /// @param reduction Reduction type (min, max or average)
  void list_timings(MPI_Comm comm, std::set<TimingType> type,
                    Table::Reduction reduction);

  /// Return timing
  /// @param[in] task The task name to retrieve the timing for
  /// @returns Values (count, total wall time, total user time, total
  /// system time) for given task.
  std::tuple<int, double, double, double> timing(std::string task);

private:
  // List of timings for tasks, map from string to (num_timings,
  // total_wall_time, total_user_time, total_system_time)
  std::map<std::string, std::tuple<int, double, double, double>> _timings;
};
} // namespace dolfinx::common
