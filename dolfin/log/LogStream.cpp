// Copyright (C) 2003-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-03-13
// Last changed: 2007-05-15

#include <stdio.h>
#include <cmath>
#include <string>
#include <dolfin/common/constants.h>
#include <dolfin/common/types.h>
#include "log.h"
#include "LogManager.h"
#include "LogStream.h"

using namespace dolfin;

// Definition of the global cout and endl variables
LogStream dolfin::cout(LogStream::COUT);
LogStream dolfin::endl(LogStream::ENDL);

//-----------------------------------------------------------------------------
LogStream::LogStream(Type type)
{
  this->type = type;
  buffer = new char[DOLFIN_LINELENGTH];
  current = 0;
}
//-----------------------------------------------------------------------------
LogStream::~LogStream()
{
  if (buffer)
    delete [] buffer;
}
//-----------------------------------------------------------------------------
LogStream& LogStream::operator<<(const char* s)
{
  add(s);
  return *this;
}
//-----------------------------------------------------------------------------
LogStream& LogStream::operator<<(const std::string& s)
{
  add(s.c_str());
  return *this;
}
//-----------------------------------------------------------------------------
LogStream& LogStream::operator<<(const LogStream& stream)
{
  if (stream.type == ENDL)
  {
    LogManager::logger.message(buffer);
    current = 0;
    buffer[0] = '\0';
  }
  else
    add(stream.buffer);
  
  return *this;
}
//-----------------------------------------------------------------------------
LogStream& LogStream::operator<<(int a)
{
  char tmp[DOLFIN_LINELENGTH];
  snprintf(tmp, DOLFIN_LINELENGTH, "%d", a);
  add(tmp);
  return *this;
}
//-----------------------------------------------------------------------------
LogStream& LogStream::operator<<(unsigned int a)
{
  char tmp[DOLFIN_LINELENGTH];
  snprintf(tmp, DOLFIN_LINELENGTH, "%u", a);
  add(tmp);
  return *this;
}
//-----------------------------------------------------------------------------
LogStream& LogStream::operator<<(double a)
{
  char tmp[DOLFIN_LINELENGTH];
  /*
    if (fabs(a) < 1e-5 || fabs(a) > 1e5)
    sprintf(tmp, "%e", a);
    else
    sprintf(tmp, "%f", a);
  */
  snprintf(tmp, DOLFIN_LINELENGTH, "%.3g", a);
  add(tmp);
  return *this;
}
//-----------------------------------------------------------------------------
LogStream& LogStream::operator<<(complex z)
{
  char tmp[DOLFIN_LINELENGTH];
  snprintf(tmp, DOLFIN_LINELENGTH, "%f + %fi", z.real(), z.imag());
  add(tmp);
  return *this;
}
//-----------------------------------------------------------------------------
void LogStream::disp() const
{
  // This is used for debugging

  printf("This i a LogStream of type ");
  switch ( type ) {
  case COUT:
    printf("cout.\n");
    break;
  default:
    printf("endl.\n");
  }

  printf("The buffer size is %d. Currently at position %d. \n",
	 DOLFIN_LINELENGTH, current);
}
//-----------------------------------------------------------------------------
void LogStream::add(const char* msg)
{
  for (int i = 0; msg[i]; i++)
  {
    if (current >= (DOLFIN_LINELENGTH-1))
    {
      LogManager::logger.message(buffer);
      current = 0;
      buffer[0] = '\0';
      return;
    }
    buffer[current++] = msg[i];
  }
  buffer[current] = '\0';
}
//-----------------------------------------------------------------------------
