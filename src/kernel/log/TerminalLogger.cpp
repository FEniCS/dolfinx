// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <iostream>
#include <stdlib.h>
#include <dolfin/TerminalLogger.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
TerminalLogger::TerminalLogger() : GenericLogger()
{

}
//-----------------------------------------------------------------------------
TerminalLogger::~TerminalLogger()
{
  
}
//-----------------------------------------------------------------------------
void TerminalLogger::info(const char* msg)
{
  std::cout << "DOLFIN: " << msg << endl;
}
//-----------------------------------------------------------------------------
void TerminalLogger::debug(const char* msg)
{
  std::cout << "DOLFIN debug: " << msg << endl;
}
//-----------------------------------------------------------------------------
void TerminalLogger::warning(const char* msg)
{
  std::cout << "DOLFIN warning: " << msg << endl;
}
//-----------------------------------------------------------------------------
void TerminalLogger::error(const char* msg)
{
  std::cout << "DOLFIN error: " << msg << endl;
  exit(1);
}
//-----------------------------------------------------------------------------
