// Copyright (C) 2003-2016 Anders Logg, 2015 Jan Blechta
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Thanks to Jim Tilander for many helpful hints.
//
// Modified by Ola Skavhaug 2007, 2009

#ifndef __LOGGER_H
#define __LOGGER_H

#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <set>
#include <thread>
#include <tuple>

#include <dolfin/common/timing.h>
#include <dolfin/common/MPI.h>
#include "Table.h"
#include "LogLevel.h"

namespace dolfin
{

  class Logger
  {
  public:

    /// Constructor
    Logger();

    /// Destructor
    ~Logger();

    /// Print message
    void log(std::string msg, int log_level=INFO) const;

    /// Print underlined message
    void log_underline(std::string msg, int log_level=INFO) const;

    /// Print warning
    void warning(std::string msg) const;

    /// Print error message and throw exception
    void error(std::string msg) const;

    /// Print error message, prefer this to the above generic error message
    void dolfin_error(std::string location,
                      std::string task,
                      std::string reason,
                      int mpi_rank=-1) const;

    /// Issue deprecation warning for removed feature
    void deprecation(std::string feature,
                     std::string version_deprecated,
                     std::string message) const;

    /// Begin task (increase indentation level)
    void begin(std::string msg, int log_level=INFO);

    /// End task (decrease indentation level)
    void end();

    /// Draw progress bar
    void progress (std::string title, double p) const;

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

    /// Register timing (for later summary)
    void register_timing(std::string task,
                         std::tuple<double, double, double> elapsed);

    /// Return a summary of timings and tasks in a Table, optionally clearing
    /// stored timings
    Table timings(TimingClear clear, std::set<TimingType> type);

    /// List a summary of timings and tasks, optionally clearing stored timings.
    /// ``MPI_AVG`` reduction is printed. Collective on ``Logger::mpi_comm()``.
    void list_timings(TimingClear clear, std::set<TimingType> type);

    /// Dump a summary of timings and tasks to XML file, optionally clearing
    /// stored timings. ``MPI_MAX``, ``MPI_MIN`` and ``MPI_AVG`` reductions are
    /// stored. Collective on ``Logger::mpi_comm()``.
    void dump_timings_to_xml(std::string filename, TimingClear clear);

    /// Return timing (count, total wall time, total user time,
    /// total system time) for given task, optionally clearing
    /// all timings for the task
    std::tuple<std::size_t, double, double, double>
      timing(std::string task, TimingClear clear);

    /// Monitor memory usage. Call this function at the start of a
    /// program to continuously monitor the memory usage of the
    /// process.
    void monitor_memory_usage();

    /// Return MPI Communicator of Logger
    MPI_Comm mpi_comm()
    { return _mpi_comm; }

    /// Helper function for reporting memory usage
    void _report_memory_usage(size_t num_mb);

    /// Helper function for dolfin_debug macro
    void __debug(std::string msg) const;

    /// Helper function for dolfin_dolfin_assert macro
    void __dolfin_assert(std::string file, unsigned long line,
                  std::string function, std::string check) const;

  private:

    // Write message
    void write(int log_level, std::string msg) const;

    // True iff logging is active
    bool _active;

    // Current log level
    int _log_level;

    // Current indentation level
    int indentation_level;

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

    // MPI Communicator
    MPI_Comm _mpi_comm;

  };

}

#endif
