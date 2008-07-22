// Copyright (C) 2003-2008 Anders Logg and Jim Tilander.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2007.
// 
// First added:  2003-03-13
// Last changed: 2008-07-21

#ifndef __LOG_H
#define __LOG_H

#include <string>
#include <map>
#include <stdarg.h>
#include <dolfin/common/types.h>

namespace dolfin
{
  
  /// The DOLFIN log system provides the following set of functions for
  /// uniform handling of log messages, warnings and errors. In addition,
  /// macros are provided for debug messages and assertions.
  ///
  /// Only messages with a debug level higher than or equal to the global
  /// debug level are printed (the default being zero). The global debug
  /// level may be controlled by
  ///
  ///    set("debug level", debug_level);
  ///
  /// where debug_level is the desired debug level.
  ///
  /// The output destination can be controlled by
  ///
  ///    set("output destination", destination);
  ///
  /// where destination is one of "terminal" (default) or "silent". Setting
  /// the output destination to "silent" means no messages will be printed.

  /// Print message
  void message(std::string msg, ...);

  /// Print message
  void message(int debug_level, std::string msg, ...);
  
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

  /// Print summary of timings and tasks, clearing stored timings
  void summary();

  /// Return timing (average) for given task
  real timing(std::string task);

  // Helper function for dolfin_debug macro
  void __debug(std::string file, unsigned long line, std::string function, std::string format, ...);

  // Helper function for dolfin_assert macro
  void __dolfin_assert(std::string file, unsigned long line, std::string function, std::string format, ...);

}

// Debug macros (with varying number of arguments)
#define dolfin_debug(msg)              do { dolfin::__debug(__FILE__, __LINE__, __FUNCTION__, msg); } while (false)
#define dolfin_debug1(msg, a0)         do { dolfin::__debug(__FILE__, __LINE__, __FUNCTION__, msg, a0); } while (false)
#define dolfin_debug2(msg, a0, a1)     do { dolfin::__debug(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1); } while (false)
#define dolfin_debug3(msg, a0, a1, a2) do { dolfin::__debug(__FILE__, __LINE__, __FUNCTION__, msg, a0, a1, a2); } while (false)

// Assertion, only active if DEBUG is defined
#ifdef DEBUG
#define dolfin_assert(check) do { if ( !(check) ) { dolfin::__dolfin_assert(__FILE__, __LINE__, __FUNCTION__, "(" #check ")"); } } while (false)
#else
#define dolfin_assert(check)
#endif

#endif
