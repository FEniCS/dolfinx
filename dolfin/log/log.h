// Copyright (C) 2003-2011 Anders Logg and Jim Tilander
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
// Modified by Ola Skavhaug 2007, 2009
//
// First added:  2003-03-13
// Last changed: 2011-10-19

#ifndef __LOG_H
#define __LOG_H

#include <string>
#include <cassert>
#include <dolfin/common/types.h>
#include "LogLevel.h"

namespace dolfin
{

  class Variable;
  class Parameters;

  /// The DOLFIN log system provides the following set of functions for
  /// uniform handling of log messages, warnings and errors. In addition,
  /// macros are provided for debug messages and assertions.
  ///
  /// Only messages with a debug level higher than or equal to the current
  /// log level are printed (the default being zero). Logging may also be
  /// turned off by calling set_log_active(false).

  /// Print message
  void info(std::string msg, ...);

  /// Print parameter (using output of str() method)
  void info(const Parameters& parameters, bool verbose=false);

  /// Print variable (using output of str() method)
  void info(const Variable& variable, bool verbose=false);

  /// Print message to stream
  void info_stream(std::ostream& out, std::string msg);

  /// Print underlined message
  void info_underline(std::string msg, ...);

  /// Print warning
  void warning(std::string msg, ...);

  /// Print error message and throw an exception
  void error(std::string msg, ...);

  /// Print error message, prefer this to the above generic error message
  void dolfin_error(std::string location,
                    std::string task,
                    std::string reason, ...);

  /// Print message at given debug level
  void log(int debug_level, std::string msg, ...);

  /// Begin task (increase indentation level)
  void begin(std::string msg, ...);

  /// Begin task (increase indentation level)
  void begin(int debug_level, std::string msg, ...);

  /// End task (decrease indentation level)
  void end();

  /// Turn logging on or off
  void set_log_active(bool active=true);

  /// Set log level
  void set_log_level(int level);

  /// Set output stream
  void set_output_stream(std::ostream& out);

  /// Get log level
  int get_log_level();

  /// List a summary of timings and tasks, optionally clearing stored timings
  void list_timings(bool reset=false);

  /// This function is deprecated, use list_timings
  void summary(bool reset=false);

  /// Return timing (average) for given task, optionally clearing timing for task
  double timing(std::string task, bool reset=false);

  /// Report that functionality has not (yet) been implemented to work in parallel
  void not_working_in_parallel(std::string what);

  /// Check value and print an informative error message if invalid
  void check_equal(uint value, uint valid_value, std::string task, std::string value_name);

  // Helper function for dolfin_debug macro
  void __debug(std::string file, unsigned long line, std::string function, std::string format, ...);

}

// The following two macros are the only "functions" in DOLFIN
// named dolfin_foo. Other functions can be placed inside the
// DOLFIN namespace and therefore don't require a prefix.

// Debug macros (with varying number of arguments)
#define dolfin_debug(msg)                  do { dolfin::__debug(__FILE__, __LINE__, __FUNCTION__, msg); } while (false)
#define dolfin_debug1(msg, a0)             do { dolfin::__debug(__FILE__, __LINE__, __FUNCTION__, msg, a0); } while (false)
#define dolfin_debug2(msg, a0, a1)         do { dolfin::__debug(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1); } while (false)
#define dolfin_debug3(msg, a0, a1, a2)     do { dolfin::__debug(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1, a2); } while (false)
#define dolfin_debug4(msg, a0, a1, a2, a3) do { dolfin::__debug(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1, a2, a3); } while (false)

// Not implemented error, reporting function name and line number
#define dolfin_not_implemented() \
  do { \
    error("The function '%s' has not been implemented (in %s line %d).", \
          __FUNCTION__, __FILE__, __LINE__); \
  } while (false)

#endif
