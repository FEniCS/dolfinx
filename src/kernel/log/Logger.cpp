// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <stdio.h>
#include <unistd.h>
#include <stdarg.h>

#include <dolfin/dolfin_settings.h>
#include <dolfin/constants.h>
#include <dolfin/TerminalLogger.h>
#include <dolfin/CursesLogger.h>
#include <dolfin/SilentLogger.h>
#include <dolfin/Logger.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Logger::Logger()
{
  // Will be initialised when first needed
  log = 0;
  
  buffer = new char[DOLFIN_LINELENGTH];
  location  = new char[DOLFIN_LINELENGTH];

  state = true;
}
//-----------------------------------------------------------------------------
Logger::~Logger()
{
  if ( log )
    delete log;
  log = 0;
  
  if ( buffer )
    delete [] buffer;
  buffer = 0;
  
  if ( location )
    delete [] location;
  location = 0;
}
//-----------------------------------------------------------------------------
void Logger::info(const char* msg)
{
  if ( !state )
    return;

  init();
  log->info(msg);
}
//-----------------------------------------------------------------------------
void Logger::info(const char* format, va_list aptr)
{
  if ( !state )
    return;

  init();

  vsprintf(buffer, format, aptr);
  log->info(buffer);
}
//-----------------------------------------------------------------------------
void Logger::debug(const char* file, unsigned long line,
		   const char* function, const char* format, ...)
{
  init();
  
  sprintf(location, "%s:%d: %s()", file, (int) line, function);
  
  va_list aptr;
  va_start(aptr, format);
  
  vsprintf(buffer, format, aptr);
  log->debug(buffer, location);
  
  va_end(aptr);
}
//-----------------------------------------------------------------------------
void Logger::warning(const char* file, unsigned long line,
		     const char* function, const char* format, ...)
{
  init();

  sprintf(location, "%s:%d: %s()", file, (int) line, function);
  
  va_list aptr;
  va_start(aptr, format);
  
  vsprintf(buffer, format, aptr);
  log->warning(buffer, location);

  va_end(aptr);
}
//-----------------------------------------------------------------------------
void Logger::error(const char* file, unsigned long line,
		   const char* function, const char* format, ...)
{
  if ( !state )
    return;

  init();

  sprintf(location, "%s:%d: %s()", file, (int) line, function);
  
  va_list aptr;
  va_start(aptr, format);
  
  vsprintf(buffer, format, aptr);
  log->error(buffer, location);
  
  va_end(aptr);
}
//-----------------------------------------------------------------------------
void Logger::dassert(const char* file, unsigned long line,
		     const char* function, const char* format, ...)
{
  init();

  sprintf(location, "%s:%d: %s()", file, (int) line, function);
  
  va_list aptr;
  va_start(aptr, format);
  
  vsprintf(buffer, format, aptr);
  log->dassert(buffer, location);
  
  va_end(aptr);
}
//-----------------------------------------------------------------------------
void Logger::progress(const char* title, const char* label, real p)
{
  if ( !state )
    return;

  init();
  log->progress(title, label, p);
}
//-----------------------------------------------------------------------------
void Logger::update()
{
  if ( !state )
    return;

  init();
  log->update();
}
//-----------------------------------------------------------------------------
void Logger::quit()
{
  init();
  log->quit();
}
//-----------------------------------------------------------------------------
bool Logger::finished()
{
  init();
  return log->finished();
}
//-----------------------------------------------------------------------------
void Logger::progress_add(Progress* p)
{
  init();
  log->progress_add(p);
}
//-----------------------------------------------------------------------------
void Logger::progress_remove (Progress *p)
{
  init();
  log->progress_remove(p);
}
//-----------------------------------------------------------------------------
void Logger::start()
{
  init();
  log->start();
}
//-----------------------------------------------------------------------------
void Logger::end()
{
  init();
  log->end();
}
//-----------------------------------------------------------------------------
void Logger::active(bool state)
{
  this->state = state;
}
//-----------------------------------------------------------------------------
void Logger::init(const char* type)
{
  // Delete old logger
  if ( log )
    delete log;
  
  // Choose output type
  if ( strcasecmp(type, "plain text") == 0 )
  {
    log = new TerminalLogger();
    return;
  }
  else if ( strcasecmp(type, "curses") )
  {
#ifdef NO_CURSES
    log = new TerminalLogger();
    dolfin_warning1("DOLFIN compiled without curses, using plain text.");
    return;
#else
    log = new CursesLogger();
    return;
#endif
  }
  else if ( strcasecmp(type, "silent") )
  {
    log = new SilentLogger();
    return;
  }
  else
  {
    log = new TerminalLogger();
    dolfin_warning1("Unknown output type \"%s\", using plain text.", type);
  }
}
//-----------------------------------------------------------------------------
void Logger::init()
{
  if ( log )
    return;

  // Default is plain text
  init("plain text");
}
//-----------------------------------------------------------------------------
