// Copyright (C) 2003-2009 Anders Logg and Jim Tilander.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2007, 2009.
//
// First added:  2003-03-13
// Last changed: 2011-01-21

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
  /// turned off by calling log(false).

  /// Print message
  void info(std::string msg, ...);

  /// Print message at given debug level
  void info(int debug_level, std::string msg, ...);

  /// Print parameter (using output of str() method)
  void info(const Parameters& parameters, bool verbose=false);

  /// Print variable (using output of str() method)
  void info(const Variable& variable, bool verbose=false);

  /// Print message to stream
  void info_stream(std::ostream& out, std::string msg);

  /// Print underlined message
  void info_underline(std:: string msg, ...);

  /// Print warning
  void warning(std::string msg, ...);

  /// Print error message and throw an exception
  void error(std::string msg, ...);

  /// Begin task (increase indentation level)
  void begin(std::string msg, ...);

  /// Begin task (increase indentation level)
  void begin(int debug_level, std::string msg, ...);

  /// End task (decrease indentation level)
  void end();

  /// Turn logging on or off
  void logging(bool active=true);

  /// Set log level
  void set_log_level(int level);

  /// Set output stream
  void set_output_stream(std::ostream& out);

  /// Get log level
  int get_log_level();

  /// Print summary of timings and tasks, optionally clearing stored timings
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

// Debug macros (with varying number of arguments)
#define dolfin_debug(msg)              do { dolfin::__debug(__FILE__, __LINE__, __FUNCTION__, msg); } while (false)
#define dolfin_debug1(msg, a0)         do { dolfin::__debug(__FILE__, __LINE__, __FUNCTION__, msg, a0); } while (false)
#define dolfin_debug2(msg, a0, a1)     do { dolfin::__debug(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1); } while (false)
#define dolfin_debug3(msg, a0, a1, a2) do { dolfin::__debug(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1, a2); } while (false)

// Not implemented error, reporting function name and line number
#define dolfin_not_implemented() \
  do { \
    error("The function '%s' has not been implemented (in %s line %d).", \
          __FUNCTION__, __FILE__, __LINE__); \
  } while (false)

#endif
