// Copyright (C) 2003-2008 Anders Logg and Jim Tilander
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
// First added:  2003-03-14
// Last changed: 2010-11-10

#ifndef __PROGRESS_H
#define __PROGRESS_H

#include <string>

namespace dolfin
{

  // FIXME: Replace implementation with wrapper for boost::progress_display
  // FIXME: See http://www.boost.org/doc/libs/1_42_0/libs/timer/timer.htm

  /// This class provides a simple way to create and update progress
  /// bars during a computation.
  ///
  /// *Example*
  ///     A progress bar may be used either in an iteration with a known number
  ///     of steps:
  ///
  /// @code{.cpp}
  ///
  ///         Progress p("Iterating...", n);
  ///         for (int i = 0; i < n; i++)
  ///         {
  ///           ...
  ///           p++;
  ///         }
  /// @endcode
  ///
  ///     or in an iteration with an unknown number of steps:
  ///
  /// @code{.cpp}
  ///
  ///         Progress p("Iterating...");
  ///         while (t < T)
  ///         {
  ///           ...
  ///           p = t / T;
  ///         }
  /// @endcode

  class Progress
  {
  public:

    /// Create progress bar with a known number of steps
    ///
    /// @param    title (std::string)
    ///         The title.
    /// @param    n (unsigned int)
    ///         Number of steps.
    Progress(std::string title, unsigned int n);

    /// Create progress bar with an unknown number of steps
    ///
    /// @param    title (std::string)
    ///         The title.
    Progress(std::string title);

    /// Destructor
    ~Progress();

    /// Set current position
    ///
    /// @param    p (double)
    ///         The position.
    void operator=(double p);

    /// Increment progress
    void operator++(int);

  private:

    // Update progress
    void update(double p);

    // Title of progress bar
    std::string _title;

    // Number of steps
    std::size_t _n;

    // Current position
    std::size_t i;

    // Minimum time increment
    double t_step;

    // Minimum counter increment
    std::size_t c_step;

    // Current progress
    double _p;

    // Time for latest update
    double _t;

    // Time for last checking the time
    double tc;

    // Always visible
    bool always;

    // Finished flag
    bool finished;

    // Displayed flag
    bool displayed;

    // Counter for updates
    std::size_t counter;

  };

}

#endif
