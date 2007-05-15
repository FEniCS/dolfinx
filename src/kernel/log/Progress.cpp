// Copyright (C) 2003-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Thanks to Jim Tilander for many helpful hints.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2003-03-14
// Last changed: 2007-05-13

#include <dolfin/utils.h>
#include <dolfin/log.h>
#include <dolfin/LogManager.h>
#include <dolfin/Progress.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Progress::Progress(std::string title, unsigned int n)
{
  if (n <= 0)
    error("Number of steps for progress session must be positive.");


  _title = title;
  p0 = 0.0;
  p1 = 0.0;

  progress_step = 0.1;

  i = 0;
  this->n = n;

  stopped = false;

  // Write first progress bar
  LogManager::logger.progress(_title, p1);
}
//-----------------------------------------------------------------------------
Progress::Progress(std::string title)
{
  _title = title;
  
  p0 = 0.0;
  p1 = 0.0;

  progress_step = 0.1;

  i = 0;
  n = 0;
  LogManager::logger.progress(_title, p1);
}
//-----------------------------------------------------------------------------
Progress::~Progress()
{
  // Step to end
  if ( p1 != 1.0 && !stopped )
  {
    p1 = 1.0;
    LogManager::logger.progress(_title, p1);
  }
}
//-----------------------------------------------------------------------------
void Progress::setStep(real step)
{
  // We can't go through the parameter system, since the parameter system
  // depends on the log system and the log system should not depend on the
  // parameter system.

  progress_step = step;
}
//-----------------------------------------------------------------------------
void Progress::operator=(unsigned int i)
{
  if ( n == 0 )
    error("Cannot specify step number for progress session with unknown number of steps.");

  p1 = checkBounds(i);
  update();  
}
//-----------------------------------------------------------------------------
void Progress::operator=(real p)
{
  if ( n != 0 )
    error("Cannot specify value for progress session with given number of steps.");

  p1 = checkBounds(p);
  update();
}
//-----------------------------------------------------------------------------
void Progress::operator++()
{
  if ( n == 0 )
    error("Cannot step progress for session with unknown number of steps.");

  if ( i < (n-1) )
    i++;

  p1 = checkBounds(i);  
  update();
}
//-----------------------------------------------------------------------------
void Progress::operator++(int)
{
  if ( n == 0 )
    error("Cannot step progress for session with unknown number of steps.");
  
  if ( i < (n-1) )
    i++;
  
  p1 = checkBounds(i);
  update();
}
//-----------------------------------------------------------------------------
void Progress::stop()
{
  stopped = true;
}
//-----------------------------------------------------------------------------
real Progress::value()
{
  return p1;
}
//-----------------------------------------------------------------------------
std::string Progress::title()
{
  return _title;
}
//-----------------------------------------------------------------------------
real Progress::checkBounds(unsigned int i)
{
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
  if ( (p1 - p0) < progress_step && (p1 != 1.0 || p1 == p0) )
    return;

  LogManager::logger.progress(_title, p1);
  p0 = p1;
}
//-----------------------------------------------------------------------------
