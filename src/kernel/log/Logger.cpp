// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <stdio.h>
#include <unistd.h>
#include <stdarg.h>

#include <dolfin/dolfin_settings.h>
#include <dolfin/constants.h>
#include <dolfin/TerminalLogger.h>
#include <dolfin/CursesLogger.h>
#include <dolfin/Logger.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Logger::Logger()
{
  // Will be initialised when first needed
  log = 0;
  
  buffer = new char[DOLFIN_LINELENGTH];
  location  = new char[DOLFIN_LINELENGTH];
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
  init();
  log->info(msg);
}
//-----------------------------------------------------------------------------
void Logger::info(const char* format, va_list aptr)
{
  init();

  vsprintf(buffer, format, aptr);
  log->info(buffer);
}
//-----------------------------------------------------------------------------
void Logger::debug(const char* file, unsigned long line,
		   const char* function, const char* format, ...)
{
  init();
  
  sprintf(location, "%s:%d: %s()", file, line, function);
  
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

  sprintf(location, "%s:%d: %s()", file, line, function);
  
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
  init();

  sprintf(location, "%s:%d: %s()", file, line, function);
  
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

  sprintf(location, "%s:%d: %s()", file, line, function);
  
  va_list aptr;
  va_start(aptr, format);
  
  vsprintf(buffer, format, aptr);
  log->dassert(buffer, location);
  
  va_end(aptr);
}
//-----------------------------------------------------------------------------
void Logger::progress(const char* title, const char* label, real p)
{
  init();
  log->progress(title, label, p);
}
//-----------------------------------------------------------------------------
void Logger::update()
{
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
void Logger::init()
{
  if ( log != 0 )
    return;

  // Get output type
  string type = dolfin_get("output");
  
  if ( type == "plain text" )
    log = new TerminalLogger();
  else if ( type == "curses" )
    log = new CursesLogger();
  else {
    log = new CursesLogger();
    dolfin_warning1("Unknown output type \"%s\".", type.c_str());
  }

}
//-----------------------------------------------------------------------------
