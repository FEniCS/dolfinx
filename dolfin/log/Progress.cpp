// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Thanks to Jim Tilander for many helpful hints.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2003-03-14
// Last changed: 2008-03-06

#include <dolfin/common/utils.h>
#include <dolfin/common/timing.h>
#include "log.h"
#include "LogManager.h"
#include "Progress.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Progress::Progress(std::string title, unsigned int n) 
  : title(title), n(n), i(0), p_step(0.1), t_step(1.0), p(0), t(0)
{
  if (n <= 0)
    error("Number of steps for progress session must be positive.");

  //LogManager::logger.progress(title, 0.0);
  t = time();
}
//-----------------------------------------------------------------------------
Progress::Progress(std::string title)
  : title(title), n(0), i(0), p_step(0.1), t_step(1.0), p(0), t(0)
{
  //LogManager::logger.progress(title, 0.0);
  t = time();
}
//-----------------------------------------------------------------------------
Progress::~Progress()
{
  if (this->p == 0.0)
    LogManager::logger.message(title + " (finished)");
  else if (this ->p < 1.0)
    LogManager::logger.progress(title, 1.0);
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
  p = std::max(std::min(p, 1.0), 0.0);
  const real t = time();
  
  const bool p_check = p - this->p >= p_step - DOLFIN_EPS;
  const bool t_check = t - this->t >= t_step - DOLFIN_EPS;

  // Only update when the increase is significant
  if ((p_check && t_check) || t_check)
  {
    LogManager::logger.progress(title, p);
    this->p = p;
    this->t = t;
  }
}
//-----------------------------------------------------------------------------
