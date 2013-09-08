// Copyright (C) 2008 Anders Logg
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
// First added:  2008-06-13
// Last changed: 2010-11-16

#ifndef __TIMER_H
#define __TIMER_H

#include <string>

namespace dolfin
{

  /// A timer can be used for timing tasks. The basic usage is
  ///
  ///   Timer timer("Assembling over cells");
  ///
  /// The timer is started at construction and timing ends
  /// when the timer is destroyed (goes out of scope). It is
  /// also possible to start and stop a timer explicitly by
  ///
  ///   timer.start();
  ///   timer.stop();
  ///
  /// Timings are stored globally and a summary may be printed
  /// by calling
  ///
  ///   list_timings();

  class Timer
  {
  public:

    /// Create timer
    Timer(std::string task);

    /// Destructor
    ~Timer();

    /// Start timer
    void start();

    /// Stop timer
    double stop();

    /// Return value of timer (or time at start if not stopped)
    double value() const;

  private:

    // Name of task
    std::string _task;

    // Start time
    double t;

    // True if timer has been stopped
    bool stopped;

  };

}

#endif
