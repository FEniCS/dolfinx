// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <stdio.h>
#include <stdarg.h>

#include <dolfin/constants.h>
#include <dolfin/TerminalLogger.h>
#include <dolfin/Logger.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Logger::Logger()
{
  log = new TerminalLogger();
  buffer = new char[DOLFIN_LINELENGTH];
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
  va_list aptr;
  va_start(aptr, format);
  
  vsprintf(buffer, format, aptr);
  log->debug(buffer);

  va_end(aptr);
}
//-----------------------------------------------------------------------------
void Logger::warning(const char* file, unsigned long line,
							const char* function, const char* format, ...)
{
  va_list aptr;
  va_start(aptr, format);
  
  vsprintf(buffer, format, aptr);
  log->warning(buffer);

  va_end(aptr);
}
//-----------------------------------------------------------------------------
void Logger::error(const char* file, unsigned long line,
						 const char* function, const char* format, ...)
{
  va_list aptr;
  va_start(aptr, format);
  
  vsprintf(buffer, format, aptr);
  log->error(buffer);

  va_end(aptr);
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
