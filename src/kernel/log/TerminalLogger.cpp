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
  std::cout << msg << std::endl;
}
//-----------------------------------------------------------------------------
void TerminalLogger::debug(const char* msg, const char* location)
{
  std::cout << "[Debug at " << location << "]: " << msg << std::endl;
}
//-----------------------------------------------------------------------------
void TerminalLogger::warning(const char* msg, const char* location)
{
  std::cout << "[Warning at " << location << "]: " << msg << std::endl;
}
//-----------------------------------------------------------------------------
void TerminalLogger::error(const char* msg, const char* location)
{
  std::cout << "[Error at " << location << "]: " << msg << std::endl;
  exit(1);
}
//-----------------------------------------------------------------------------
void TerminalLogger::dassert(const char* msg, const char* location)
{
  std::cout << "[Assertion " << msg << " failed at " << location << "]: " << msg << std::endl;
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
void TerminalLogger::update()
{
  // FIXME: Maybe we should flush the output?
}
//-----------------------------------------------------------------------------
void TerminalLogger::quit()
{
  // FIXME: What should be done here?
}
//-----------------------------------------------------------------------------
bool TerminalLogger::finished()
{
  return false;
}
//-----------------------------------------------------------------------------
void TerminalLogger::progress_add(Progress* p)
{
  // Do nothing here
}
//-----------------------------------------------------------------------------
void TerminalLogger::progress_remove(Progress *p)
{
  // Do nothing here
}
//-----------------------------------------------------------------------------
