// Copyright (C) 2003-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Thanks to Jim Tilander for many helpful hints.
//
// Modified by Ola Skavhaug, 2007, 2009.
//
// First added:  2003-03-13
// Last changed: 2009-05-08

#include <stdarg.h>
#include <stdio.h>
#include <signal.h>
#include <sstream>
#include <dolfin/common/types.h>
#include <dolfin/common/constants.h>
#include <dolfin/common/Variable.h>
#include "LogManager.h"
#include "log.h"

using namespace dolfin;

// Buffer
static unsigned int buffer_size = 0;
static char* buffer = 0;

// Buffer allocation
void allocate_buffer(std::string msg)
{
  // va_list, start, end require a char pointer of fixed size so we
  // need to allocate the buffer here. We allocate twice the size of
  // the format string and at least DOLFIN_LINELENGTH. This should be
  // ok in most cases.

  unsigned int new_size = std::max(2*msg.size(), static_cast<uint>(DOLFIN_LINELENGTH));
  if (new_size > buffer_size)
  {
    delete [] buffer;
    buffer = new char[new_size];
    buffer_size = new_size;
  }
}

// Macro for parsing arguments
#define read(buffer, msg) \
  allocate_buffer(msg); \
  va_list aptr; \
  va_start(aptr, msg); \
  vsnprintf(buffer, buffer_size, msg.c_str(), aptr); \
  va_end(aptr);

//-----------------------------------------------------------------------------
void dolfin::info(std::string msg, ...)
{
  read(buffer, msg);
  LogManager::logger.info(buffer);
}
//-----------------------------------------------------------------------------
void dolfin::info(int debug_level, std::string msg, ...)
{
  read(buffer, msg);
  LogManager::logger.info(buffer, debug_level);
}
//-----------------------------------------------------------------------------
void dolfin::info(const Variable& variable)
{
  info(variable.str());
}
//-----------------------------------------------------------------------------
void dolfin::info_stream(std::ostream& out, std::string msg)
{
  std::ostream& old_out = LogManager::logger.get_output_destination();
  LogManager::logger.set_output_destination(out);
  LogManager::logger.info(msg);
  LogManager::logger.set_output_destination(old_out);
}
//-----------------------------------------------------------------------------
void dolfin::info_underline(std:: string msg, ...)
{
  read(buffer, msg);
  LogManager::logger.info_underline(buffer);
}
//-----------------------------------------------------------------------------
void dolfin::warning(std::string msg, ...)
{
  read(buffer, msg);
  LogManager::logger.warning(buffer);
}
//-----------------------------------------------------------------------------
void dolfin::error(std::string msg, ...)
{
  read(buffer, msg);
  LogManager::logger.error(buffer);
}
//-----------------------------------------------------------------------------
void dolfin::begin(std::string msg, ...)
{
  read(buffer, msg);
  LogManager::logger.begin(buffer);
}
//-----------------------------------------------------------------------------
void dolfin::begin(int debug_level, std::string msg, ...)
{
  read(buffer, msg);
  LogManager::logger.begin(buffer, debug_level);
}
//-----------------------------------------------------------------------------
void dolfin::end()
{
  LogManager::logger.end();
}
//-----------------------------------------------------------------------------
std::string dolfin::indent(std::string s)
{
  const std::string indentation("  ");
  std::stringstream is;
  is << indentation;
  for (uint i = 0; i < s.size(); ++i)
  {
    is << s[i];
    if (s[i] == '\n') // && i < s.size() - 1)
      is << indentation;
  }

  return is.str();
}
//-----------------------------------------------------------------------------
void dolfin::summary(bool reset)
{
  LogManager::logger.summary(reset);
}
//-----------------------------------------------------------------------------
double dolfin::timing(std::string task, bool reset)
{
  return LogManager::logger.timing(task, reset);
}
//-----------------------------------------------------------------------------
void dolfin::__debug(std::string file, unsigned long line,
                     std::string function, std::string format, ...)
{
  read(buffer, format);
  std::ostringstream ost;
  ost << file << ":" << line << " in " << function << "()";
  std::string msg = std::string(buffer) + " [at " + ost.str() + "]";
  LogManager::logger.__debug(msg);
}
//-----------------------------------------------------------------------------
void dolfin::__dolfin_assert(std::string file, unsigned long line,
                      std::string function, std::string format, ...)
{
  read(buffer, format);
  std::ostringstream ost;
  ost << file << ":" << line << " in " << function << "()";
  std::string msg = std::string(buffer) + " [at " + ost.str() + "]";
  LogManager::logger.__assert(msg);
}
//-----------------------------------------------------------------------------
