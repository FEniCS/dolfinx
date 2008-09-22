// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-06-13
// Last changed: 2008-07-23

#ifndef __TIMER_H
#define __TIMER_H

#include <iostream>

#include <dolfin/parameter/parameters.h>
#include <dolfin/log/LogManager.h>
#include "timing.h"

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
  ///   summary();

  class Timer
  {
  public:

    /// Create timer
    Timer(std::string task) : task(""), t(time()), stopped(false)
    { const std::string prefix = dolfin_get("timer prefix"); this->task = prefix + task; }

    /// Destructor
    ~Timer()
    { if (!stopped) stop(); }

    /// Start timer
    inline void start()
    { t = time(); stopped = false; }
    
    /// Stop timer
    void stop()
    { LogManager::logger.registerTiming(task, time() - t); stopped = true; }

  private:

    // Name of task
    std::string task;

    // Start time
    real t;

    // True if timer has been stopped
    bool stopped;

  };

}

#endif
