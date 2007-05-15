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

using namespace dolfin;

// Buffers
char buffer0[DOLFIN_LINELENGTH];
char buffer1[DOLFIN_LINELENGTH];

//-----------------------------------------------------------------------------
void dolfin::message(std::string msg, ...)
{
  va_list aptr;
  va_start(aptr, msg);
  vsnprintf(buffer0, DOLFIN_LINELENGTH, msg.c_str(), aptr);
  va_end(aptr);
  LogManager::logger.message(std::string(buffer0));
}
//-----------------------------------------------------------------------------
void dolfin::message(int debug_level, std::string msg, ...)
{
  va_list aptr;
  va_start(aptr, msg);
  vsnprintf(buffer0, DOLFIN_LINELENGTH, msg.c_str(), aptr);
  va_end(aptr);
  LogManager::logger.message(std::string(buffer0), debug_level);
}
//-----------------------------------------------------------------------------
void dolfin::warning(std::string msg, ...)
{
  va_list aptr;
  va_start(aptr, msg);
  vsnprintf(buffer0, DOLFIN_LINELENGTH, msg.c_str(), aptr);
  va_end(aptr);
  LogManager::logger.warning(std::string(buffer0));
}
//-----------------------------------------------------------------------------
void dolfin::error(std::string msg, ...)
{
  va_list aptr;
  va_start(aptr, msg);
  vsnprintf(buffer0, DOLFIN_LINELENGTH, msg.c_str(), aptr);
  va_end(aptr);
  LogManager::logger.error(std::string(buffer0));
}
//-----------------------------------------------------------------------------
void dolfin::begin(std::string msg, ...)
{
  va_list aptr;
  va_start(aptr, msg);
  vsnprintf(buffer0, DOLFIN_LINELENGTH, msg.c_str(), aptr);
  va_end(aptr);
  LogManager::logger.begin(std::string(buffer0));
}
//-----------------------------------------------------------------------------
void dolfin::begin(int debug_level, std::string msg, ...)
{
  va_list aptr;
  va_start(aptr, msg);
  vsnprintf(buffer0, DOLFIN_LINELENGTH, msg.c_str(), aptr);
  va_end(aptr);
  LogManager::logger.begin(std::string(buffer0), debug_level);
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
  va_list aptr;
  va_start(aptr, format);
  vsnprintf(buffer0, DOLFIN_LINELENGTH, format.c_str(), aptr);
  snprintf(buffer1, DOLFIN_LINELENGTH, "%s:%d: %s()", file.c_str(), (int) line, function.c_str());
  va_end(aptr);
  std::string msg = std::string(buffer0) + "[at " + std::string(buffer1) + "]";
  LogManager::logger.__debug(msg);
}
//-----------------------------------------------------------------------------
void dolfin::__assert(std::string file, unsigned long line,
                      std::string function, std::string format, ...)
{
  va_list aptr;
  va_start(aptr, format);
  vsnprintf(buffer0, DOLFIN_LINELENGTH, format.c_str(), aptr);
  snprintf(buffer1, DOLFIN_LINELENGTH, "%s:%d: %s()", file.c_str(), (int) line, function.c_str());
  va_end(aptr);
  std::string msg = std::string(buffer0) + " [at " + std::string(buffer1) + "]";
  LogManager::logger.__assert(msg);
}
//-----------------------------------------------------------------------------
