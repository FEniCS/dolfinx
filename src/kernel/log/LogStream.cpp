// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <stdio.h>

#include <dolfin/constants.h>
#include <dolfin/LoggerMacros.h>
#include <dolfin/LogManager.h>
#include <dolfin/LogStream.h>

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
  if ( buffer )
	 delete [] buffer;
  buffer = 0;
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
  if ( stream.type == ENDL ) {
    LogManager::log.info(buffer);
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
  char tmp[DOLFIN_WORDLENGTH];
  sprintf(tmp, "%d", a);
  add(tmp);
  return *this;
}
//-----------------------------------------------------------------------------
LogStream& LogStream::operator<<(real a)
{
  char tmp[DOLFIN_WORDLENGTH];
  sprintf(tmp, "%f", a);
  add(tmp);
  return *this;
}
//-----------------------------------------------------------------------------
void LogStream::add(const char* msg)
{
  for (int i = 0; msg[i]; i++) {
    if ( current >= (DOLFIN_LINELENGTH-1) ) {
      LogManager::log.info(buffer);
      current = 0;
      buffer[0] = '\0';
      return;
    }
    buffer[current++] = msg[i];
  }
  buffer[current] = '\0';
}
//-----------------------------------------------------------------------------
