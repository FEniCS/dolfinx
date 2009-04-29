// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Thanks to Jim Tilander for many helpful hints.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2003-03-14
// Last changed: 2008-06-20

#include <dolfin/common/constants.h>
#include <dolfin/common/timing.h>
#include "log.h"
#include "LogManager.h"
#include "Progress.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Progress::Progress(std::string title, unsigned int n)
  : title(title), n(n), i(0), p_step(0.1), t_step(1.0), p(0), t(0),
    always(false), finished(false), displayed(false)
{
  if (n <= 0)
    error("Number of steps for progress session must be positive.");

  //LogManager::logger.progress(title, 0.0);
  t = time();

  //When "debug level" is more than 0, progress is always visible
  if (LogManager::logger.getDebugLevel() > 0 )
    always = true;

}

//-----------------------------------------------------------------------------
Progress::Progress(std::string title)
  : title(title), n(0), i(0), p_step(0.1), t_step(1.0), p(0), t(0),
    always(false), finished(false), displayed(false)
{
  //LogManager::logger.progress(title, 0.0);
  t = time();

  //When "debug level" is more than 0, progress is always visible
  if (LogManager::logger.getDebugLevel() > 0 )
    always = true;

}
//-----------------------------------------------------------------------------
Progress::~Progress()
{
}
//-----------------------------------------------------------------------------
void Progress::operator=(double p)
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

  update(static_cast<double>(i) / static_cast<double>(n));
}
//-----------------------------------------------------------------------------
void Progress::update(double p)
{
  //p = std::max(std::min(p, 1.0), 0.0);
  //const bool p_check = p - this->p >= p_step - DOLFIN_EPS;

  const double t = time();
  const bool t_check = t - this->t >= t_step - DOLFIN_EPS;

  // Only update when the increase is significant
  if (t_check || always || (p >= 1.0 && displayed && !finished))
  {
    LogManager::logger.progress(title, p);
    this->p = p;
    this->t = t;
    always = false;
    displayed = true;
  }

  // Update finished flag
  if (p >= 1.0)
    finished = true;

}
//-----------------------------------------------------------------------------
