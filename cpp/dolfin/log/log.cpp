// Copyright (C) 2003-2016 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "log.h"
#include "LogManager.h"
#include <cstdarg>
#include <cstdlib>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Variable.h>
#include <dolfin/common/constants.h>
#include <dolfin/parameter/Parameters.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdarg.h>
#include <stdio.h>

using namespace dolfin;

static std::unique_ptr<char[]> buffer;
static unsigned int buffer_size = 0;

// Buffer allocation
void allocate_buffer(std::string msg)
{
  // va_list, start, end require a char pointer of fixed size so we
  // need to allocate the buffer here. We allocate twice the size of
  // the format string and at least DOLFIN_LINELENGTH. This should be
  // ok in most cases.
  unsigned int new_size
      = std::max(static_cast<unsigned int>(2 * msg.size()),
                 static_cast<unsigned int>(DOLFIN_LINELENGTH));
  // static_cast<unsigned int>(DOLFIN_LINELENGTH));
  if (new_size > buffer_size)
  {
    buffer.reset(new char[new_size]);
    buffer_size = new_size;
  }
}

// Macro for parsing arguments
#define read(buffer, msg)                                                      \
  allocate_buffer(msg);                                                        \
  va_list aptr;                                                                \
  va_start(aptr, msg);                                                         \
  vsnprintf(buffer, buffer_size, msg.c_str(), aptr);                           \
  va_end(aptr);

//-----------------------------------------------------------------------------
void dolfin::info(std::string msg, ...)
{
  if (!LogManager::logger().is_active())
    return; // optimization
  read(buffer.get(), msg);
  LogManager::logger().log(buffer.get());
}
//-----------------------------------------------------------------------------
void dolfin::info(const common::Variable& variable, bool verbose)
{
  if (!LogManager::logger().is_active())
    return; // optimization
  info(variable.str(verbose));
}
//-----------------------------------------------------------------------------
void dolfin::info(const dolfin::parameter::Parameters& parameters, bool verbose)
{
  // Need separate function for Parameters since we can't make Parameters
  // a subclass of Variable (gives cyclic dependencies)

  if (!LogManager::logger().is_active())
    return; // optimization
  info(parameters.str(verbose));
}
//-----------------------------------------------------------------------------
void dolfin::info_stream(std::ostream& out, std::string msg)
{
  if (!LogManager::logger().is_active())
    return; // optimization
  std::ostream& old_out = LogManager::logger().get_output_stream();
  LogManager::logger().set_output_stream(out);
  LogManager::logger().log(msg);
  LogManager::logger().set_output_stream(old_out);
}
//-----------------------------------------------------------------------------
void dolfin::info_underline(std::string msg, ...)
{
  if (!LogManager::logger().is_active())
    return; // optimization
  read(buffer.get(), msg);
  LogManager::logger().log_underline(buffer.get());
}
//-----------------------------------------------------------------------------
void dolfin::warning(std::string msg, ...)
{
  if (!LogManager::logger().is_active())
    return; // optimization
  read(buffer.get(), msg);
  LogManager::logger().warning(buffer.get());
}
//-----------------------------------------------------------------------------
void dolfin::error(std::string msg, ...)
{
  read(buffer.get(), msg);
  LogManager::logger().error(buffer.get());
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_error(std::string location, std::string task,
                          std::string reason, ...)
{
  read(buffer.get(), reason);
  LogManager::logger().dolfin_error(location, task, buffer.get());
}
//-----------------------------------------------------------------------------
void dolfin::deprecation(std::string feature, std::string version_deprecated,
                         std::string message, ...)
{
  read(buffer.get(), message);
  LogManager::logger().deprecation(feature, version_deprecated, buffer.get());
}
//-----------------------------------------------------------------------------
void dolfin::log(int log_level, std::string msg, ...)
{
  if (!LogManager::logger().is_active())
    return; // optimization
  read(buffer.get(), msg);
  LogManager::logger().log(buffer.get(), log_level);
}
//-----------------------------------------------------------------------------
void dolfin::begin(std::string msg, ...)
{
  if (!LogManager::logger().is_active())
    return; // optimization
  read(buffer.get(), msg);
  LogManager::logger().begin(buffer.get());
}
//-----------------------------------------------------------------------------
void dolfin::begin(int log_level, std::string msg, ...)
{
  if (!LogManager::logger().is_active())
    return; // optimization
  read(buffer.get(), msg);
  LogManager::logger().begin(buffer.get(), log_level);
}
//-----------------------------------------------------------------------------
void dolfin::end()
{
  if (!LogManager::logger().is_active())
    return; // optimization
  LogManager::logger().end();
}
//-----------------------------------------------------------------------------
void dolfin::set_log_active(bool active)
{
  LogManager::logger().set_log_active(active);
}
//-----------------------------------------------------------------------------
void dolfin::set_log_level(int level)
{
  LogManager::logger().set_log_level(level);
}
//-----------------------------------------------------------------------------
void dolfin::set_indentation_level(std::size_t indentation_level)
{
  LogManager::logger().set_indentation_level(indentation_level);
}
//-----------------------------------------------------------------------------
void dolfin::set_output_stream(std::ostream& out)
{
  LogManager::logger().set_output_stream(out);
}
//-----------------------------------------------------------------------------
int dolfin::get_log_level() { return LogManager::logger().get_log_level(); }
//-----------------------------------------------------------------------------
void dolfin::monitor_memory_usage()
{
  LogManager::logger().monitor_memory_usage();
}
//-----------------------------------------------------------------------------
void dolfin::not_working_in_parallel(std::string what)
{
  if (MPI::size(MPI_COMM_WORLD) > 1)
  {
    dolfin_error("log.cpp", "perform operation in parallel",
                 "%s is not yet working in parallel.\n"
                 "***          Consider filing a bug report at %s",
                 what.c_str(),
                 "https://bitbucket.org/fenics-project/dolfin/issues");
  }
}
//-----------------------------------------------------------------------------
void dolfin::__debug(std::string file, unsigned long line, std::string function,
                     std::string format, ...)
{
  read(buffer.get(), format);
  std::ostringstream ost;
  ost << "[at " << file << ":" << line << " in " << function << "()]";
  LogManager::logger().__debug(ost.str());
  std::string msg = std::string(buffer.get());
  LogManager::logger().__debug(msg);
}
//-----------------------------------------------------------------------------
void dolfin::__dolfin_assert(std::string file, unsigned long line,
                             std::string function, std::string check)
{
  LogManager::logger().__dolfin_assert(file, line, function, check);
}
//-----------------------------------------------------------------------------
