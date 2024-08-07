// Copyright (C) 2008 Anders Logg, 2015 Jan Blechta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <boost/timer/timer.hpp>
#include <optional>
#include <string>

namespace dolfinx::common
{

/// A timer can be used for timing tasks. The basic usage is
///
///   Timer timer("Assembling over cells");
///
/// The timer is started at construction and timing ends when the timer
/// is destroyed (goes out of scope). It is also possible to start and
/// stop a timer explicitly by
///
///   timer.start(); timer.stop();
///
/// Timings are stored globally and a summary may be printed by calling
///
///   list_timings();

class Timer
{
public:
  /// Create timer
  /// if a task name is provided this enables logging, otherwise without logging
  Timer(std::optional<std::string> task = std::nullopt);

  /// Destructor
  ~Timer();

  /// Zero and start timer
  void start();

  /// Resume timer. Not well-defined for logging timer
  void resume();

  /// Stop timer, return wall time elapsed and store timing data into
  /// logger
  double stop();

  /// Return wall, user and system time in seconds
  std::array<double, 3> elapsed() const;

private:
  // Name of task
  std::string _task;

  // Implementation of timer
  boost::timer::cpu_timer _timer;
};
} // namespace dolfinx::common
