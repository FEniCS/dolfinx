// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Thanks to Jim Tilander for many helpful hints.
//
// Modified by Ola Skavhaug, 2007.
//
// First added:  2003-03-13
// Last changed: 2008-06-13

#include <stdarg.h>
#include <stdio.h>
#include <signal.h>
#include <dolfin/common/types.h>
#include <dolfin/common/constants.h>
#include "LogManager.h"
#include "log.h"

using namespace dolfin;

// Buffers
static char buffer[DOLFIN_LINELENGTH];

#define read(buffer, msg) \
  va_list aptr; \
  va_start(aptr, msg); \
  vsnprintf(buffer, DOLFIN_LINELENGTH, msg.c_str(), aptr); \
  va_end(aptr);

//-----------------------------------------------------------------------------
void dolfin::message(std::string msg, ...)
{
  read(buffer, msg);
  LogManager::logger.message(buffer);
}
//-----------------------------------------------------------------------------
void dolfin::message(int debug_level, std::string msg, ...)
{
  read(buffer, msg);
  LogManager::logger.message(buffer, debug_level);
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
void dolfin::summary()
{
  LogManager::logger.summary();
}
//-----------------------------------------------------------------------------
const std::map<std::string, std::pair<dolfin::uint, dolfin::real> >& timings()
{
  return LogManager::logger.timings();
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
