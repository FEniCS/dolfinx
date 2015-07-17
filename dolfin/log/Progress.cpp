// Copyright (C) 2003-2011 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Thanks to Jim Tilander for many helpful hints.
//
// Modified by Garth N. Wells, 2006.
// Modified by Ola Skavhaug, 2009.
//
// First added:  2003-03-14
// Last changed: 2011-11-14

#include <dolfin/common/constants.h>
#include <dolfin/common/timing.h>
#include "log.h"
#include "LogManager.h"
#include "Progress.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Progress::Progress(std::string title, unsigned int n)
  : _title(title), _n(n), i(0), t_step(0.5), c_step(1), _p(0), _t(0), tc(0),
    always(false), finished(false), displayed(false), counter(0)
{
  if (n <= 0)
  {
    dolfin_error("Progress.cpp",
                 "create progress bar",
                 "Number of steps for progress bar must be positive");
  }

  // LogManager::logger.progress(title, 0.0);
  _t = time();

  // When log level is TRACE or lower, always display at least the 100% message
  if (LogManager::logger().get_log_level() <= TRACE )
    always = true;
}
//-----------------------------------------------------------------------------
Progress::Progress(std::string title)
  : _title(title), _n(0), i(0), t_step(0.5), c_step(1), _p(0), _t(0), tc(0),
    always(false), finished(false), displayed(false), counter(0)
{
  // LogManager::logger.progress(title, 0.0);
  _t = time();

  // When log level is TRACE or lower, always display at least the 100% message
  if (LogManager::logger().get_log_level() <= TRACE )
    always = true;
}
//-----------------------------------------------------------------------------
Progress::~Progress()
{
  // Display last progress bar if not displayed
  if (displayed && !finished)
    LogManager::logger().progress(_title, 1.0);
}
//-----------------------------------------------------------------------------
void Progress::operator=(double p)
{
  if (_n != 0)
  {
    dolfin_error("Progress.cpp",
                 "update progress bar",
                 "Cannot specify value for progress bar with given number of steps");
  }

  update(p);
}
//-----------------------------------------------------------------------------
void Progress::operator++(int)
{
  if (_n == 0)
  {
    dolfin_error("Progress.cpp",
                 "increment progress bar",
                 "Cannot step progress bar for session with unknown number of steps");
  }

  if (i < _n)
    i++;

  update(static_cast<double>(i) / static_cast<double>(_n));
}
//-----------------------------------------------------------------------------
void Progress::update(double p)
{
  // FIXME: We should be able to simplify this...

  // Check that enough number of updates have passed so we don't call
  // time() to often which is costly
  if (counter++ < c_step)
    return;
  counter = 0;

  // Check if we have already finished
  if (finished)
    return;

  // Check that enough time has passed since last update
  const double t = time();
  const bool t_check = (t - _t >= t_step - DOLFIN_EPS);

  // Adjust counter step
  const double fraction = 0.1;
  if (t - this->tc < fraction * t_step)
    c_step *= 2;
  else if (t - this->tc > t_step && c_step >= 2)
    c_step /= 2;
  this->tc = t;

  // Reset time for next update
  if (t_check)
  {
    _p = p;
    _t = t;
  }

  // Assume we want to update the progress
  bool do_log_update = t_check;

  // Nontrivial check for first update
  if (t_check && !always && !displayed && p >= 0.7)
  {
    // skip the first update, since it will probably reach 100%
    // before the next time t_check is true
    do_log_update = false;

    // ...but don't skip the next (pretend we displayed this one)
    displayed = true;
  }

  // Nontrivial check for last update
  if (p >= 1.0)
  {
    // always display 100% message if a message has already been shown
    if (displayed || always)
      do_log_update = true;

    // ...but don't show more than one
    finished = true;
  }

  // Only update when the increase is significant
  if (do_log_update)
  {
    LogManager::logger().progress(_title, p);
    displayed = true;
  }
}
//-----------------------------------------------------------------------------
