// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <dolfin/utils.h>
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
void TerminalLogger::debug(const char* msg, const char* location)
{
  std::cout << "DOLFIN debug [" << location << "]: " << msg << endl;
}
//-----------------------------------------------------------------------------
void TerminalLogger::warning(const char* msg, const char* location)
{
  std::cout << "DOLFIN warning [" << location << "]: " << msg << endl;
}
//-----------------------------------------------------------------------------
void TerminalLogger::error(const char* msg, const char* location)
{
  std::cout << "DOLFIN error [" << location << "]: " << msg << endl;
  exit(1);
}
//-----------------------------------------------------------------------------
void TerminalLogger::progress(const char* title, const char* label, real p)
{
  int N = DOLFIN_TERM_WIDTH - 15;
  int n = (int) (p*((double) N));
  
  // Print the title
  printf("| %s", title);
  for (int i = 0; i < (N-length(title)-1); i++)
	 printf(" ");
  printf("|\n");
  
  // Print the progress bar
  printf("|");
  for (int i = 0; i < n; i++)
	 printf("=");
  if ( n > 0 && n < N ) {
	 printf("|");
	 n++;
  }
  for (int i = n; i < N; i++)
	 printf("-");
  printf("| %.1f\%\n", 100.0*p);
}
//-----------------------------------------------------------------------------
