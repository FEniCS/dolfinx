// Copyright (C) 2003-2011 Anders Logg
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
// Modified by Ola Skavhaug, 2007, 2009.
//
// First added:  2003-03-13
// Last changed: 2011-04-11

#ifndef __LOGGER_H
#define __LOGGER_H

#include <map>
#include <ostream>
#include <string>
#include <dolfin/common/types.h>
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
                      std::string reason) const;

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
    inline bool is_active() { return active; }

    /// Set log level
    void set_log_level(int log_level);

    /// Get log level
    inline int get_log_level() const { return log_level; }

    /// Register timing (for later summary)
    void register_timing(std::string task, double elapsed_time);

    /// Print summary of timings and tasks, optionally clearing stored timings
    void summary(bool reset=false);

    /// Return timing (average) for given task, optionally clearing timing for task
    double timing(std::string task, bool reset=false);

    /// Helper function for dolfin_debug macro
    void __debug(std::string msg) const;

  private:

    // Write message
    void write(int log_level, std::string msg) const;

    // True iff logging is active
    bool active;

    // Current log level
    int log_level;

    // Current indentation level
    int indentation_level;

    // Optional stream for logging
    std::ostream* logstream;

    // List of timings for tasks, map from string to (num_timings, total_time)
    std::map<std::string, std::pair<uint, double> > timings;

  };

}

#endif
