// Copyright (C) 2003-2016 Anders Logg, 2015 Jan Blechta
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "LogLevel.h"
#include "Table.h"
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

namespace log
{

/// Handling of error messages, logging and informational display

class Logger
{
public:
  /// Constructor
  Logger();

  /// Destructor
  ~Logger();

  /// Print message
  void log(std::string msg, int log_level = INFO) const;

  /// Print underlined message
  void log_underline(std::string msg, int log_level = INFO) const;

  /// Print warning
  void warning(std::string msg) const;

  /// Print error message and throw exception
  void error(std::string msg) const;

  /// Print error message, prefer this to the above generic error
  /// message
  void dolfin_error(std::string location, std::string task, std::string reason,
                    int mpi_rank = -1) const;

  /// Issue deprecation warning for removed feature
  void deprecation(std::string feature, std::string version_deprecated,
                   std::string message) const;

  /// Begin task (increase indentation level)
  void begin(std::string msg, int log_level = INFO);

  /// End task (decrease indentation level)
  void end();

  /// Draw progress bar
  void progress(std::string title, double p) const;

  /// Set output stream
  void set_output_stream(std::ostream& stream);

  /// Get output stream
  std::ostream& get_output_stream() { return *logstream; }

  /// Turn logging on or off
  void set_log_active(bool active);

  /// Return true iff logging is active
  inline bool is_active() { return _active; }

  /// Set log level
  void set_log_level(int log_level);

  /// Get log level
  inline int get_log_level() const { return _log_level; }

  /// Set indentation level
  void set_indentation_level(std::size_t indentation_level);

  /// Register timing (for later summary)
  void register_timing(std::string task,
                       std::tuple<double, double, double> elapsed);

  /// Return a summary of timings and tasks in a Table, optionally
  /// clearing stored timings
  Table timings(TimingClear clear, std::set<TimingType> type);

  /// List a summary of timings and tasks, optionally clearing
  /// stored timings.  ``MPI_AVG`` reduction is printed. Collective
  /// on ``Logger::mpi_comm()``.
  void list_timings(TimingClear clear, std::set<TimingType> type);

  /// Return timing (count, total wall time, total user time, total
  /// system time) for given task, optionally clearing all timings
  /// for the task
  std::tuple<std::size_t, double, double, double> timing(std::string task,
                                                         TimingClear clear);

  /// Monitor memory usage. Call this function at the start of a
  /// program to continuously monitor the memory usage of the
  /// process.
  void monitor_memory_usage();

  /// Return MPI Communicator of Logger
  MPI_Comm mpi_comm() { return _mpi_comm; }

  /// Helper function for reporting memory usage
  void _report_memory_usage(size_t num_mb);

  /// Helper function for dolfin_debug macro
  void __debug(std::string msg) const;

private:
  // Write message
  void write(int log_level, std::string msg) const;

  // True iff logging is active
  bool _active;

  // Current log level
  int _log_level;

  // Current indentation level
  int _indentation_level;

  // Optional stream for logging
  std::ostream* logstream;

  // List of timings for tasks, map from string to
  // (num_timings, total_wall_time, total_user_time, total_system_time)
  std::map<std::string, std::tuple<std::size_t, double, double, double>>
      _timings;

  // Thread used for monitoring memory usage
  std::unique_ptr<std::thread> _thread_monitor_memory_usage;

  // Maximum memory usage so far
  long int _maximum_memory_usage;

  // Map for stringifying TimingType
  static std::map<TimingType, std::string> _TimingType_descr;

  // FIXME: This should be a dolfin::MPI::Comm, the MPI-awareness of
  //        the logging needs to be fixed.
  // MPI Communicator
  MPI_Comm _mpi_comm;
};
}
}