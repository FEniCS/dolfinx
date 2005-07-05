// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Thanks to Jim Tilander for many helpful hints.
//
// First added:  2003-03-13
// Last changed: 2005



#include <stdarg.h>
#include <signal.h>
#include <dolfin/LoggerMacros.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void dolfin::dolfin_info(const char *msg, ...)
{
  va_list aptr;
  va_start(aptr, msg);

  LogManager::log.info(msg, aptr);

  va_end(aptr);
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_info_aptr(const char *msg, va_list aptr)
{
  LogManager::log.info(msg, aptr);
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_update()
{
  LogManager::log.update();
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_quit()
{
  LogManager::log.quit();
}
//-----------------------------------------------------------------------------
bool dolfin::dolfin_finished()
{
  return LogManager::log.finished();
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_segfault()
{
  dolfin_info("Deliberately raising a segmentation fault (so you can attach a debugger).");
  raise(SIGSEGV);
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_start()
{
  LogManager::log.start();
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_start(const char* msg, ...)
{
  va_list aptr;
  va_start(aptr, msg);

  LogManager::log.info(msg, aptr);

  va_end(aptr);

  LogManager::log.start();
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_end()
{
  LogManager::log.end();
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_end(const char* msg, ...)
{
  LogManager::log.end();

  va_list aptr;
  va_start(aptr, msg);

  LogManager::log.info(msg, aptr);

  va_end(aptr);
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_output(const char* type)
{
  LogManager::log.init(type);
}
//-----------------------------------------------------------------------------
void dolfin::dolfin_log(bool state)
{
  LogManager::log.active(state);
}
//-----------------------------------------------------------------------------
