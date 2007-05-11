// Copyright (C) 2003-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-03-13
// Last changed: 2007-05-11

#include <string>
#include <iostream>
#include <stdexcept>

#include <stdio.h>
#include <stdarg.h>

#include <dolfin/constants.h>
#include <dolfin/Logger.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Logger::Logger()
  : destination(terminal), state(true), debug_level(0), indentation_level(0),
    buffer0(0), buffer1(0)
{
  // Initialize buffers
  buffer0 = new char[DOLFIN_LINELENGTH];
  buffer1 = new char[DOLFIN_LINELENGTH];
}
//-----------------------------------------------------------------------------
Logger::~Logger()
{
  if (buffer0)
    delete [] buffer0;

  if (buffer1)
    delete [] buffer1;
}
//-----------------------------------------------------------------------------
void Logger::info(const char* msg)
{
  if (!state)
    return;

  // Write message
  std::string s(msg);
  write(0, s);
}
//-----------------------------------------------------------------------------
void Logger::info(const char* format, va_list aptr)
{
  if (!state || 0 > this->debug_level)
    return;

  vsprintf(buffer0, format, aptr);

  // Write message
  std::string s(buffer0);
  write(0, s);
}
//-----------------------------------------------------------------------------
void Logger::info(int debug_level, const char* format, va_list aptr)
{
  if (!state || debug_level > this->debug_level)
    return;

  vsprintf(buffer0, format, aptr);

  // Write message
  std::string s(buffer0);
  write(debug_level, s);
}
//-----------------------------------------------------------------------------
void Logger::debug(const char* file, unsigned long line,
		   const char* function, const char* format, ...)
{
  va_list aptr;
  va_start(aptr, format);

  vsprintf(buffer0, format, aptr);
  sprintf(buffer1, "%s:%d: %s()", file, (int) line, function);
  
  va_end(aptr);

  // Write message
  std::string s = std::string("Debug: ") + std::string(buffer0) + " [at " + std::string(buffer1) + "]";
  write(0, s);
}
//-----------------------------------------------------------------------------
void Logger::warning(const char* file, unsigned long line,
		     const char* function, const char* format, ...)
{
  va_list aptr;
  va_start(aptr, format);

  vsprintf(buffer0, format, aptr);
  sprintf(buffer1, "%s:%d: %s()", file, (int) line, function);

  va_end(aptr);

  // Write message
  std::string s = std::string("*** Warning: ") + std::string(buffer0) + " [at " + std::string(buffer1) + "]";
  write(0, s);
}
//-----------------------------------------------------------------------------
void Logger::error(const char* file, unsigned long line,
		   const char* function, const char* format, ...)
{
  va_list aptr;
  va_start(aptr, format);

  vsprintf(buffer0, format, aptr);
  sprintf(buffer1, "%s:%d: %s()", file, (int) line, function);  

  va_end(aptr);

  // Throw exception
  std::string s = std::string("*** Error: ") + std::string(buffer0) + " [at " + std::string(buffer1) + "]";
  throw std::runtime_error(s);
}
//-----------------------------------------------------------------------------
void Logger::dassert(const char* file, unsigned long line,
		     const char* function, const char* format, ...)
{
  va_list aptr;
  va_start(aptr, format);

  vsprintf(buffer0, format, aptr);
  sprintf(buffer1, "%s:%d: %s()", file, (int) line, function);

  va_end(aptr);

  // Throw exception
  std::string s = std::string("*** Assertion ") + std::string(buffer0) + " failed [at " + std::string(buffer1) + "]";
  throw std::runtime_error(s);
}
//-----------------------------------------------------------------------------
void Logger::progress(const char* title, const char* label, real p)
{
  if ( !state )
    return;

  int N = DOLFIN_TERM_WIDTH - 15;
  int n = static_cast<int>(p*static_cast<real>(N));
  
  // Print the title
  std::string s = "| " + std::string(title);
  for (uint i = 0; i < (N - std::string(title).size() - 1); i++)
    s += " ";
  s += "|";
  write(0, s);
  
  // Print the progress bar
  s = "|";
  for (int i = 0; i < n; i++)
    s += "=";
  if (n > 0 && n < N)
  {
    s += "|";
    n++;
  }
  for (int i = n; i < N; i++)
    s += "-";
  s += "| ";
  sprintf(buffer0, "%.1f%%", 100.0*p);
  s += std::string(buffer0);
  write(0, s);
}
//-----------------------------------------------------------------------------
void Logger::begin()
{
  indentation_level++;
}
//-----------------------------------------------------------------------------
void Logger::end()
{
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
  // Choose output destination
  if ( strcmp(destination, "terminal") == 0 )
    this->destination = terminal;
  else if ( strcmp(destination, "silent") == 0 )
    this->destination = silent;
  else
  {
    this->destination = terminal;
    info("Unknown output destination, using plain text.");
  }
}
//-----------------------------------------------------------------------------
void Logger::write(int debug_level, std::string msg)
{
  // Check if we should produce output
  if (!state || debug_level > this->debug_level)
    return;

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
