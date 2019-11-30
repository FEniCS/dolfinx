// Copyright (C) 2003-2016 Anders Logg, 2015 Jan Blechta
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/common/MPI.h>
#include <dolfin/common/Table.h>
#include <dolfin/common/timing.h>
#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <thread>
#include <tuple>

namespace dolfin
{

namespace common
{

/// Timer logging

class TimeLogger
{
public:
  /// Constructor
  TimeLogger();

  /// Destructor
  ~TimeLogger() = default;

  /// Register timing (for later summary)
  void register_timing(std::string task, double wall, double user,
                       double system);

  /// Return a summary of timings and tasks in a Table
  Table timings(std::set<TimingType> type);

  /// List a summary of timings and tasks. ``MPI_AVG`` reduction is
  /// printed.
  /// @param mpi_comm MPI Communicator
  /// @param type Set of possible timings: wall, user or system
  void list_timings(MPI_Comm mpi_comm, std::set<TimingType> type);

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
} // namespace common
} // namespace dolfin
