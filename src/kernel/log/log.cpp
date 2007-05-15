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
#include <dolfin/constants.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void dolfin::message(std::string msg, ...)
{
  char buffer[DOLFIN_LINELENGTH];
  va_list aptr;
  va_start(aptr, msg);
  vsprintf(buffer, msg.c_str(), aptr);
  va_end(aptr);
  LogManager::logger.message(std::string(buffer));
}
//-----------------------------------------------------------------------------
void dolfin::message(int debug_level, std::string msg, ...)
{
  char buffer[DOLFIN_LINELENGTH];
  va_list aptr;
  va_start(aptr, msg);
  vsprintf(buffer, msg.c_str(), aptr);
  va_end(aptr);
  LogManager::logger.message(std::string(buffer), debug_level);
}
//-----------------------------------------------------------------------------
void dolfin::warning(std::string msg, ...)
{
  char buffer[DOLFIN_LINELENGTH];
  va_list aptr;
  va_start(aptr, msg);
  vsprintf(buffer, msg.c_str(), aptr);
  va_end(aptr);
  LogManager::logger.warning(std::string(buffer));
}
//-----------------------------------------------------------------------------
void dolfin::error(std::string msg, ...)
{
  char buffer[DOLFIN_LINELENGTH];
  va_list aptr;
  va_start(aptr, msg);
  vsprintf(buffer, msg.c_str(), aptr);
  va_end(aptr);
  LogManager::logger.error(std::string(buffer));
}
//-----------------------------------------------------------------------------
void dolfin::begin(std::string msg, ...)
{
  char buffer[DOLFIN_LINELENGTH];
  va_list aptr;
  va_start(aptr, msg);
  vsprintf(buffer, msg.c_str(), aptr);
  va_end(aptr);
  LogManager::logger.begin(std::string(buffer));
}
//-----------------------------------------------------------------------------
void dolfin::begin(int debug_level, std::string msg, ...)
{
  char buffer[DOLFIN_LINELENGTH];
  va_list aptr;
  va_start(aptr, msg);
  vsprintf(buffer, msg.c_str(), aptr);
  va_end(aptr);
  LogManager::logger.begin(std::string(buffer), debug_level);
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
  char buffer0[DOLFIN_LINELENGTH];
  char buffer1[DOLFIN_LINELENGTH];
  va_list aptr;
  va_start(aptr, format);
  vsprintf(buffer0, format.c_str(), aptr);
  sprintf(buffer1, "%s:%d: %s()", file.c_str(), (int) line, function.c_str());
  va_end(aptr);
  std::string msg = std::string(buffer0) + "[at " + std::string(buffer1) + "]";
  LogManager::logger.__debug(msg);
}
//-----------------------------------------------------------------------------
void dolfin::__assert(std::string file, unsigned long line,
                      std::string function, std::string format, ...)
{
  char buffer0[DOLFIN_LINELENGTH];
  char buffer1[DOLFIN_LINELENGTH];
  va_list aptr;
  va_start(aptr, format);
  vsprintf(buffer0, format.c_str(), aptr);
  sprintf(buffer1, "%s:%d: %s()", file.c_str(), (int) line, function.c_str());
  va_end(aptr);
  std::string msg = std::string(buffer0) + " [at " + std::string(buffer1) + "]";
  LogManager::logger.__assert(msg);
}
//-----------------------------------------------------------------------------
