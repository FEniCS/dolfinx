// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <stdio.h>
#include <unistd.h>
#include <stdarg.h>

#include <dolfin/constants.h>
#include <dolfin/TerminalLogger.h>
#include <dolfin/CursesLogger.h>
#include <dolfin/Logger.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Logger::Logger()
{
  //log = new TerminalLogger();
  log = new CursesLogger();
  
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
  log->info(msg);
}
//-----------------------------------------------------------------------------
void Logger::info(const char* format, va_list aptr)
{
  vsprintf(buffer, format, aptr);
  log->info(buffer);
}
//-----------------------------------------------------------------------------
void Logger::debug(const char* file, unsigned long line,
						 const char* function, const char* format, ...)
{
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
  sprintf(location, "%s:%d: %s()", file, line, function);
  
  va_list aptr;
  va_start(aptr, format);
  
  vsprintf(buffer, format, aptr);
  log->error(buffer, location);

  va_end(aptr);
}
//-----------------------------------------------------------------------------
void Logger::progress(const char* title, const char* label, real p)
{
  log->progress(title, label, p);
}
//-----------------------------------------------------------------------------
void Logger::update()
{
  log->update();
}
//-----------------------------------------------------------------------------
bool Logger::finished()
{
  return log->finished();
}
//-----------------------------------------------------------------------------
void Logger::progress_add(Progress* p)
{
  log->progress_add(p);
}
//-----------------------------------------------------------------------------
void Logger::progress_remove (Progress *p)
{
  log->progress_remove(p);
}
//-----------------------------------------------------------------------------
void Logger::start()
{
  log->start();
}
//-----------------------------------------------------------------------------
void Logger::end()
{
  log->end();
}
//-----------------------------------------------------------------------------
