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
// Last changed: 2011-11-14

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

  /// Print error message and throw an exception.
  /// Note to developers: this function should not be used internally
  /// in DOLFIN. Use the more informative dolfin_error instead.
  void error(std::string msg, ...);

  /// Print error message. Prefer this to the above generic error message.
  ///
  /// *Arguments*
  ///     location (std::string)
  ///         Name of the file from which the error message was generated.
  ///     task (std::string)
  ///         Name of the task that failed.
  ///         Note that this string should begin with lowercase.
  ///         Note that this string should not be punctuated.
  ///     reason (std::string)
  ///         A format string explaining the reason for the failure.
  ///         Note that this string should begin with uppercase.
  ///         Note that this string should not be punctuated.
  ///         Note that this string may contain printf style formatting.
  ///     ... (primitive types like int, uint, double, bool)
  ///         Optional arguments for the format string.
  ///
  /// Some rules of thumb:
  ///
  /// * The 'task' string should be sufficiently high level ("assemble form")
  ///   to make sense to a user.
  /// * Use the same 'task' string from all errors originating from the same
  ///   function.
  /// * The 'task' string should provide details of which particular algorithm
  ///   or method that was used ("assemble form using OpenMP assembler").
  /// * The 'reason' string should try to explain why the task failed in the
  ///   context of the task that failed ("subdomains are not yet handled").
  /// * Write "inialize mesh function" rather than "initialize MeshFunction".
  ///
  /// Some examples:
  ///
  /// dolfin_error("DofMap.cpp",
  ///              "create mapping of degrees of freedom",
  ///              "Mesh is not ordered according to the UFC numbering convention. "
  ///              "Consider calling mesh.order()");
  ///
  /// dolfin_error("File.cpp",
  ///              "open file",
  ///              "Could not create directory \"%s\"",
  ///              path.parent_path().string().c_str());
  ///
  /// dolfin_error("TriangleCell.cpp",
  ///              "access number of entities of triangle cell",
  ///              "Illegal topological dimension (%d)", dim);
  ///
  /// dolfin_error("VTKFile.cpp",
  ///              "Create VTK file",
  ///              "Unknown encoding (\"%s\"). "
  ///              "Known encodings are \"ascii\", \"base64\" and \"compressed\"",
  ///              encoding.c_str());
  ///
  /// dolfin_error("SubSystemsManager.cpp",
  ///              "initialize PETSc subsystem",
  ///              "DOLFIN has not been configured with PETSc support");
  ///
  /// dolfin_error("PETScKrylovSolver.cpp",
  ///              "Unable to solve linear system with PETSc Krylov solver",
  ///              "Matrix does not have a nonzero number of rows and columns");
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
