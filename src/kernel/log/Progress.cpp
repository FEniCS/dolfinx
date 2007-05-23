// Copyright (C) 2003-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Thanks to Jim Tilander for many helpful hints.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2003-03-14
// Last changed: 2007-05-18

#include <dolfin/utils.h>
#include <dolfin/log.h>
#include <dolfin/LogManager.h>
#include <dolfin/Progress.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Progress::Progress(std::string title, unsigned int n) 
  : title(title), n(n), i(0), step(0.1), p(0)
{
  if (n <= 0)
    error("Number of steps for progress session must be positive.");

  LogManager::logger.progress(title, 0.0);
}
//-----------------------------------------------------------------------------
Progress::Progress(std::string title)
  : title(title), n(0), i(0), step(0.1), p(0)
{
  LogManager::logger.progress(title, 0.0);
}
//-----------------------------------------------------------------------------
Progress::~Progress()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Progress::operator=(real p)
{
  if (n != 0)
    error("Cannot specify value for progress session with given number of steps.");

  update(p);
}
//-----------------------------------------------------------------------------
void Progress::operator++(int)
{
  if (n == 0)
    error("Cannot step progress for session with unknown number of steps.");
  
  if (i < n)
    i++;

  update(static_cast<real>(i) / static_cast<real>(n));
}
//-----------------------------------------------------------------------------
void Progress::update(real p)
{
  p = std::min(p, 1.0);
  p = std::max(p, 0.0);

  // Only update when the increase is significant
  if (p - this->p >= step - DOLFIN_EPS || (p >= 1.0 - DOLFIN_EPS) && p > this->p)
  {
    LogManager::logger.progress(title, p);
    this->p = p;
  }
}
//-----------------------------------------------------------------------------
