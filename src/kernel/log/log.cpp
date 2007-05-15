// Copyright (C) 2003-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Thanks to Jim Tilander for many helpful hints.
//
// Modified by Ola Skavhaug, 2007.
//
// First added:  2003-03-13
// Last changed: 2007-05-15

#include <stdarg.h>
#include <signal.h>
#include <dolfin/LogManager.h>
#include <dolfin/log.h>
#include <dolfin/constants.h>

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
void dolfin::__debug(std::string file, unsigned long line,
                     std::string function, std::string format, ...)
{
  read(buffer, format);
  std::ostringstream ost(file);
  ost << ":" << line << ": " << function;
  std::string msg = std::string(buffer) + " [at " + ost.str() + "]";
  LogManager::logger.__debug(msg);
}
//-----------------------------------------------------------------------------
void dolfin::__assert(std::string file, unsigned long line,
                      std::string function, std::string format, ...)
{
  read(buffer, format);
  std::ostringstream ost(file);
  ost << ":" << line << ": " << function;
  std::string msg = std::string(buffer) + " [at " + ost.str() + "]";
  LogManager::logger.__assert(msg);
}
//-----------------------------------------------------------------------------
