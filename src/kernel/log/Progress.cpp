// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Thanks to Jim Tilander for many helpful hints.

#include <stdio.h>

#include <dolfin/utils.h>
#include <dolfin/LogManager.h>
#include <dolfin/LoggerMacros.h>
#include <dolfin/Progress.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Progress::Progress(const char* title, int n)
{
  if ( n <= 0 )
    dolfin_error("Number of steps for progress session must be positive.");

  _title = new char[DOLFIN_WORDLENGTH];
  _label = new char[DOLFIN_WORDLENGTH];

  sprintf(_title, "%s", title);
  sprintf(_label, "%s", "");

  p0 = 0.0;
  p1 = 0.0;
  
  i = 0;
  this->n = n;

  // Notify that we have created a new progress bar
  LogManager::log.progress_add(this);
}
//-----------------------------------------------------------------------------
Progress::Progress(const char* title)
{
  _title = new char[DOLFIN_WORDLENGTH];
  _label = new char[DOLFIN_WORDLENGTH];
  
  sprintf(_title, "%s", title);
  sprintf(_label, "%s", "");

  p0 = 0.0;
  p1 = 0.0;

  i = 0;
  n = 0;

  // Notify that we have created a new progress bar
  LogManager::log.progress_add(this);
}
//-----------------------------------------------------------------------------
Progress::~Progress()
{
  // Notify that the progress bar has finished
  LogManager::log.progress_remove(this);
  
  if ( _title )
    delete [] _title;
  _title = 0;

  if ( _label )
	 delete [] _label;
  _label = 0;
}
//-----------------------------------------------------------------------------
void Progress::operator=(int i)
{
  if ( n == 0 )
	 dolfin_error("Cannot specify step number for progress session with unknown number of steps.");

  p1 = checkBounds(i);
  update();  
}
//-----------------------------------------------------------------------------
void Progress::operator=(real p)
{
  if ( n != 0 )
	 dolfin_error("Cannot specify value for progress session with given number of steps.");

  p1 = checkBounds(p);
  update();
}
//-----------------------------------------------------------------------------
void Progress::operator++()
{
  if ( n == 0 )
	 dolfin_error("Cannot step progress for session with unknown number of steps.");

  if ( i < (n-1) )
	 i++;

  p1 = checkBounds(i);  
  update();
}
//-----------------------------------------------------------------------------
void Progress::operator++(int)
{
  if ( n == 0 )
	 dolfin_error("Cannot step progress for session with unknown number of steps.");
  
  if ( i < (n-1) )
	 i++;

  p1 = checkBounds(i);
  update();
}
//-----------------------------------------------------------------------------
void Progress::update(int i, const char* format, ...)
{
  if ( n == 0 )
	 dolfin_error("Cannot specify step number for progress session with unknown number of steps.");

  va_list aptr;
  va_start(aptr, format);
  vsprintf(_label, format, aptr);
  va_end(aptr);

  p1 = checkBounds(i);
  update();
}
//-----------------------------------------------------------------------------
void Progress::update(real p, const char* format, ...)
{
  if ( n != 0 )
	 dolfin_error("Cannot specify value for progress session with given number of steps.");

  va_list aptr;
  va_start(aptr, format);
  vsprintf(_label, format, aptr);
  va_end(aptr);
  
  p1 = checkBounds(p);
  update();
}
//-----------------------------------------------------------------------------
real Progress::value()
{
  return p1;
}
//-----------------------------------------------------------------------------
const char* Progress::title()
{
  return _title;
}
//-----------------------------------------------------------------------------
const char* Progress::label()
{
  return _label;
}
//-----------------------------------------------------------------------------
real Progress::checkBounds(int i)
{
  if ( i < 0 )
    return 0.0;
  if ( i >= (n-1) )
    return 1.0;
  return ((real) i) / ((real) n);
}
//-----------------------------------------------------------------------------
real Progress::checkBounds(real p)
{
  if ( p > 1.0 )
    return 1.0;
  if ( p < 0.0 )
    return 0.0;
  return p;
}
//-----------------------------------------------------------------------------
void Progress::update()
{
  // Only update when the increase is significant
  if ( (p1 - p0) < DOLFIN_PROGRESS_STEP )
    return;

  LogManager::log.progress(_title, _label, p1);
  p0 = p1;
}
//-----------------------------------------------------------------------------
