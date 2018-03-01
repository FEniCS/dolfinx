// Copyright (C) 2008 Anders Logg, 2015 Jan Blechta
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <boost/timer/timer.hpp>
#include <string>
#include <tuple>

namespace dolfin
{

namespace common
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
  /// Create timer without logging
  Timer();

  /// Create timer with logging
  Timer(std::string task);

  /// Destructor
  ~Timer();

  /// Zero and start timer
  void start();

  /// Resume timer. Not well-defined for logging timer
  void resume();

  /// Stop timer, return wall time elapsed and store timing data
  /// into logger
  double stop();

  /// Return wall, user and system time in seconds. Wall-clock time
  /// has precision around 1 microsecond; user and system around
  /// 10 millisecond.
  std::tuple<double, double, double> elapsed() const;

private:
  // Name of task
  std::string _task;

  // Implementation of timer
  boost::timer::cpu_timer _timer;
};
}
}
