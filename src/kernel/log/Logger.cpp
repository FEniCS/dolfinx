// Copyright (C) 2003-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-03-13
// Last changed: 2007-05-11

#include <string>
#include <iostream>

#include <stdio.h>
#include <stdarg.h>

#include <dolfin/constants.h>
#include <dolfin/TerminalLogger.h>
#include <dolfin/SilentLogger.h>
#include <dolfin/Logger.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Logger::Logger()
  : destination(terminal), log(0), buffer0(0), buffer1(0), state(true),
    debug_level(0), indentation_level(0)
{
  // Initialize buffers
  buffer0 = new char[DOLFIN_LINELENGTH];
  buffer1 = new char[DOLFIN_LINELENGTH];
}
//-----------------------------------------------------------------------------
Logger::~Logger()
{
  if (log)
    delete log;
  
  if (buffer0)
    delete [] buffer0;

  if (buffer1)
    delete [] buffer1;
}
//-----------------------------------------------------------------------------
void Logger::info(const char* msg)
{
  if ( !state )
    return;

  init();

  //std::string s(msg);
  //write(s);
  
  log->info(msg);
}
//-----------------------------------------------------------------------------
void Logger::info(const char* format, va_list aptr)
{
  if (!state || 0 > this->debug_level)
    return;

  init();

  vsprintf(buffer0, format, aptr);
  log->info(buffer0);
}
//-----------------------------------------------------------------------------
void Logger::info(int debug_level, const char* format, va_list aptr)
{
  if (!state || debug_level > this->debug_level)
    return;

  init();

  vsprintf(buffer0, format, aptr);
  log->info(buffer0);
}
//-----------------------------------------------------------------------------
void Logger::debug(const char* file, unsigned long line,
		   const char* function, const char* format, ...)
{
  init();
  
  sprintf(buffer1, "%s:%d: %s()", file, (int) line, function);
  
  va_list aptr;
  va_start(aptr, format);
  
  vsprintf(buffer0, format, aptr);
  log->debug(buffer0, buffer1);
  
  va_end(aptr);
}
//-----------------------------------------------------------------------------
void Logger::warning(const char* file, unsigned long line,
		     const char* function, const char* format, ...)
{
  init();

  sprintf(buffer1, "%s:%d: %s()", file, (int) line, function);
  
  va_list aptr;
  va_start(aptr, format);
  
  vsprintf(buffer0, format, aptr);
  log->warning(buffer0, buffer1);

  va_end(aptr);
}
//-----------------------------------------------------------------------------
void Logger::error(const char* file, unsigned long line,
		   const char* function, const char* format, ...)
{
  init();

  sprintf(buffer1, "%s:%d: %s()", file, (int) line, function);
  
  va_list aptr;
  va_start(aptr, format);
  
  vsprintf(buffer0, format, aptr);
  log->error(buffer0, buffer1);
  
  va_end(aptr);
}
//-----------------------------------------------------------------------------
void Logger::dassert(const char* file, unsigned long line,
		     const char* function, const char* format, ...)
{
  init();

  sprintf(buffer1, "%s:%d: %s()", file, (int) line, function);
  
  va_list aptr;
  va_start(aptr, format);
  
  vsprintf(buffer0, format, aptr);
  log->dassert(buffer0, buffer1);
  
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
void Logger::begin()
{
  init();
  log->begin();

  indentation_level++;
}
//-----------------------------------------------------------------------------
void Logger::end()
{
  init();
  log->end();

  indentation_level--;
}
//-----------------------------------------------------------------------------
void Logger::active(bool state)
{
  this->state = state;
}
//-----------------------------------------------------------------------------
void Logger::level(int debug_level)
{
  this->debug_level = debug_level;
}
//-----------------------------------------------------------------------------
void Logger::init(const char* destination)
{
  // Delete old logger
  if ( log )
    delete log;
  
  // Choose output destination
  if ( strcmp(destination, "plain text") == 0 )
  {
    log = new TerminalLogger();
    return;
  }
  else if ( strcmp(destination, "silent") == 0 )
  {
    log = new SilentLogger();
    return;
  }
  else
  {
    log = new TerminalLogger();
    log->info("Unknown output destination, using plain text.");
  }
}
//-----------------------------------------------------------------------------
void Logger::init()
{
  if (log)
    return;

  // Default is plain text
  init("plain text");
}
//----------------------------------------------------------------------------
void Logger::write(std::string msg)
{
  // Add indentation
  for (int i = 0; i < indentation_level; i++)
    msg = "  " + msg;
  
  // Choose destination
  switch (destination)
  {
  case terminal:
    std::cout << msg << std::endl;
    break;
  case stream:
    std::cout << "Destination stream not available. Fix this Ola! :-)" << std::endl;
    break;
  default:
    // Do nothing if destination == silent
    do {} while (false);
  }
}
//----------------------------------------------------------------------------
