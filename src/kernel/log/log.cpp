// Copyright (C) 2003-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Thanks to Jim Tilander for many helpful hints.
//
// Modified by Ola Skavhaug, 2007.
//
// First added:  2003-03-13
// Last changed: 2007-05-11

#include <stdarg.h>
#include <signal.h>
#include <dolfin/LogManager.h>
#include <dolfin/log.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void dolfin::dolfin_info(const char *msg, ...)
{
  char buffer[DOLFIN_LINELENGTH];
  va_list aptr;
  va_start(aptr, msg);
  vsprintf(buffer, msg, aptr);
  va_end(aptr);
  LogManager::log.info(std::string(buffer));
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_info(int debug_level, const char *msg, ...)
{
  char buffer[DOLFIN_LINELENGTH];
  va_list aptr;
  va_start(aptr, msg);
  vsprintf(buffer, msg, aptr);
  va_end(aptr);
  LogManager::log.info(std::string(buffer), debug_level);
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_info_aptr(const char *msg, va_list aptr)
{
  char buffer[DOLFIN_LINELENGTH];
  vsprintf(buffer, msg, aptr);
  LogManager::log.info(std::string(buffer));
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_warning(const char *msg, ...)
{
  char buffer[DOLFIN_LINELENGTH];
  va_list aptr;
  va_start(aptr, msg);
  vsprintf(buffer, msg, aptr);
  va_end(aptr);
  LogManager::log.warning(std::string(buffer));
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_error(const char *msg, ...)
{
  char buffer[DOLFIN_LINELENGTH];
  va_list aptr;
  va_start(aptr, msg);
  vsprintf(buffer, msg, aptr);
  va_end(aptr);
  LogManager::log.warning(std::string(buffer));
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_begin()
{
  LogManager::log.begin();
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_begin(const char* msg, ...)
{
  char buffer[DOLFIN_LINELENGTH];
  va_list aptr;
  va_start(aptr, msg);
  vsprintf(buffer, msg, aptr);
  va_end(aptr);

  LogManager::log.info(std::string(buffer));
  LogManager::log.begin();
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_begin(int debug_level, const char* msg, ...)
{
  char buffer[DOLFIN_LINELENGTH];
  va_list aptr;
  va_start(aptr, msg);
  vsprintf(buffer, msg, aptr);
  va_end(aptr);

  LogManager::log.info(std::string(buffer), debug_level);
  LogManager::log.begin();
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_end()
{
  LogManager::log.end();
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_log(const char* destination)
{
  LogManager::log.init(destination);
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_log(bool state)
{
  LogManager::log.active(state);
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_log(int debug_level)
{
  LogManager::log.level(debug_level);
}
//-----------------------------------------------------------------------------
void dolfin::debug(const char* file, unsigned long line, const char* function, const char* format, ...)
{
  char buffer0[DOLFIN_LINELENGTH];
  char buffer1[DOLFIN_LINELENGTH];
  va_list aptr;
  va_start(aptr, format);

  vsprintf(buffer0, format, aptr);
  sprintf(buffer1, "%s:%d: %s()", file, (int) line, function);
  
  va_end(aptr);

  // Write message
  std::string msg      = std::string(buffer0);
  std::string location = " [at " + std::string(buffer1) + "]";
  LogManager::log.debug(msg, location);
}
//-----------------------------------------------------------------------------
void dolfin::dassert(const char* file, unsigned long line,
		     const char* function, const char* format, ...)
{
  char buffer0[DOLFIN_LINELENGTH];
  char buffer1[DOLFIN_LINELENGTH];
  va_list aptr;
  va_start(aptr, format);

  vsprintf(buffer0, format, aptr);
  sprintf(buffer1, "%s:%d: %s()", file, (int) line, function);

  va_end(aptr);

  // Write message
  std::string msg      = std::string(buffer0);
  std::string location = " [at " + std::string(buffer1) + "]";
  LogManager::log.dassert(msg, location);
}
//-----------------------------------------------------------------------------
