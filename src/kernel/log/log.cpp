// Copyright (C) 2003-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Thanks to Jim Tilander for many helpful hints.
//
// First added:  2003-03-13
// Last changed: 2007-05-11

#include <stdarg.h>
#include <signal.h>
#include <dolfin/log.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void dolfin::dolfin_info(const char *msg, ...)
{
  va_list aptr;
  va_start(aptr, msg);

  LogManager::log.info(0, msg, aptr);

  va_end(aptr);
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_info(int debug_level, const char *msg, ...)
{
  va_list aptr;
  va_start(aptr, msg);

  LogManager::log.info(debug_level, msg, aptr);

  va_end(aptr);
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_info_aptr(const char *msg, va_list aptr)
{
  LogManager::log.info(msg, aptr);
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_begin()
{
  LogManager::log.begin();
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_begin(const char* msg, ...)
{
  va_list aptr;
  va_start(aptr, msg);

  LogManager::log.info(0, msg, aptr);

  va_end(aptr);

  LogManager::log.begin();
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_begin(int debug_level, const char* msg, ...)
{
  va_list aptr;
  va_start(aptr, msg);

  LogManager::log.info(debug_level, msg, aptr);

  va_end(aptr);

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
