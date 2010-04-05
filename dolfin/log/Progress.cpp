// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Thanks to Jim Tilander for many helpful hints.
//
// Modified by Garth N. Wells, 2006.
// Modified by Ola Skavhaug, 2009.
//
// First added:  2003-03-14
// Last changed: 2009-04-30

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

  // LogManager::logger.progress(title, 0.0);
  t = time();

  // When log level is TRACE or lower, always display at least the 100% message
  if (LogManager::logger.get_log_level() <= TRACE )
    always = true;
}
//-----------------------------------------------------------------------------
Progress::Progress(std::string title)
  : title(title), n(0), i(0), p_step(0.1), t_step(1.0), p(0), t(0),
    always(false), finished(false), displayed(false)
{
  // LogManager::logger.progress(title, 0.0);
  t = time();

  // When log level is TRACE or lower, always display at least the 100% message
  if (LogManager::logger.get_log_level() <= TRACE )
    always = true;
}
//-----------------------------------------------------------------------------
Progress::~Progress()
{
  // Do nothing
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
  if (finished)
    return;

  //p = std::max(std::min(p, 1.0), 0.0);
  //const bool p_check = p - this->p >= p_step - DOLFIN_EPS;

  const double t = time();
  const bool t_check = (t - this->t >= t_step - DOLFIN_EPS);
  if (t_check) {
    // reset time for next update
    this->p = p;
    this->t = t;
  }

  bool do_log_update = t_check;

  if (t_check && !always && !displayed && p >= 0.7) {
    // skip the first update, since it will probably reach 100%
    // before the next time t_check is true
    do_log_update = false;

    // ...but don't skip the next (pretend we displayed this one)
    displayed = true;
  }

  if (p >= 1.0) {
    // always display 100% message if a message has already been shown
    if (displayed || always)
      do_log_update = true;

    // ...but don't show more than one
    finished = true;
  }

  // Only update when the increase is significant
  if (do_log_update)
  {
    LogManager::logger.progress(title, p);
    displayed = true;
  }
}
//-----------------------------------------------------------------------------
