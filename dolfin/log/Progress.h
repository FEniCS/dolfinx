// Copyright (C) 2003-2008 Anders Logg and Jim Tilander.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-03-14
// Last changed: 2008-03-06

#ifndef __PROGRESS_H
#define __PROGRESS_H

#include <string>
#include <stdarg.h>

#include <dolfin/common/types.h>

namespace dolfin
{
  
  /// This class provides a simple way to create and update progress
  /// bars during a computation. A progress bar may be used either
  /// in an iteration with a known number of steps:
  ///
  ///     Progress p("Iterating...", n);
  ///     for (int i = 0; i < n; i++)
  ///     {
  ///       ...
  ///       p++;
  ///     }
  ///
  /// or in an iteration with an unknown number of steps:
  ///
  ///     Progress p("Iterating...");
  ///     while (t < T)
  ///     {
  ///       ...
  ///       p = t / T;
  ///     }

  class Progress
  {
  public:

    /// Create progress bar with a known number of steps
    Progress(std::string title, unsigned int n);

    /// Create progress bar with an unknown number of steps
    Progress(std::string title);

    /// Destructor
    ~Progress();
    
    /// Set current position
    void operator=(double p);

    /// Increment progress
    void operator++(int);
    
  private:
    
    // Update progress
    void update(double p);
    
    // Title of progress bar
    std::string title;
    
    // Number of steps
    uint n;

    // Current position
    uint i;

    // Minimum progress increment
    double p_step;

    // Minimum time increment
    double t_step;

    // Current progress
    double p;
    
    // Current time
    double t;
    
  };
  
}

#endif
