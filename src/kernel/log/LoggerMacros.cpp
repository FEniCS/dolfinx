// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Thanks to Jim Tilander for many helpful hints.

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
  dolfin_info("Deliberately raising a segmentation fault.");
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
