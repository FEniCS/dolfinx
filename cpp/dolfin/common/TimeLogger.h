// Copyright (C) 2003-2016 Anders Logg, 2015 Jan Blechta
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/common/Table.h>
#include <dolfin/common/MPI.h>
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
  ~TimeLogger() {};

  /// Register timing (for later summary)
  void register_timing(std::string task,
                       std::tuple<double, double, double> elapsed);

  /// Return a summary of timings and tasks in a Table
  Table timings(std::set<TimingType> type);

  /// List a summary of timings and tasks.
  /// ``MPI_AVG`` reduction is printed. Collective on ``Logger::mpi_comm()``.
  void list_timings(std::set<TimingType> type);

  /// Return timing (count, total wall time, total user time, total
  /// system time) for given task.
  std::tuple<std::size_t, double, double, double> timing(std::string task);

  /// Return MPI Communicator of TimeLogger
  MPI_Comm mpi_comm() { return _mpi_comm; }

private:

  // List of timings for tasks, map from string to
  // (num_timings, total_wall_time, total_user_time, total_system_time)
  std::map<std::string, std::tuple<std::size_t, double, double, double>>
      _timings;

  // Map for stringifying TimingType
  static std::map<TimingType, std::string> _TimingType_descr;

  // MPI Communicator
  MPI_Comm _mpi_comm;
};
}
}
