// Copyright (C) 2008-2015 Anders Logg, Jan Blechta, Paul T. Kühner and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "TimeLogManager.h"
#include <chrono>
#include <optional>
#include <string>

namespace dolfinx::common
{
/// @brief Timer for timing tasks.
///
/// The basic usage is
///
///   Timer timer("Assembling over cells");
///
/// The timer is started at construction and timing ends when the timer
/// is destroyed (goes out of scope). It is also possible to start and
/// stop a timer explicitly by
///
///   timer.start(); timer.stop();
///
/// Timings are stored globally (in seconds) and a summary may be
/// printed by calling
///
///   list_timings();
template <typename T = std::chrono::high_resolution_clock>
class Timer
{
public:
  /// @brief Create and start timer.
  ///
  /// Time is optionally registered in the logger (in seconds).
  ///
  /// @param[in] task Name to use for registering the time in the
  /// logger. If no name is set, the timing is not registered in the
  /// logger.
  Timer(std::optional<std::string> task = std::nullopt) : _task(task) {}

  /// Destructor. If timer is still running, it is stopped and timing is
  /// registered in the logger.
  ~Timer()
  {
    if (_start_time.has_value())
      this->stop();
  }

  /// Zero and (re-)start timer.
  void start()
  {
    _acc = std::chrono::duration<double>();
    _start_time = T::now();
  }

  /// @brief Elapsed time since time has been started.
  /// @return Elapsed time duration.
  std::chrono::duration<double> elapsed() const
  {
    if (_start_time.has_value()) // Timer is running
      return T::now() - _start_time.value() + _acc;
    else // Timer is stoped
      return _acc;
  }

  /// @brief Stop timer and return elapsed (wall) time.
  ///
  /// Also registers timing data in the logger.
  ///
  /// @return The elapsed time.
  std::chrono::duration<double> stop()
  {
    if (_start_time.has_value()) // Timer is running
    {
      _acc += T::now() - _start_time.value();
      _start_time = std::nullopt;

      if (_task.has_value())
        TimeLogManager::logger().register_timing(_task.value(), _acc.count());
    }

    return _acc;
  }

private:
  // Name of task
  std::optional<std::string> _task;

  // Elapsed time offset
  std::chrono::duration<double> _acc;

  // Store start time. std::nullopt if timer has been stopped.
  std::optional<typename T::time_point> _start_time = T::now();
};
} // namespace dolfinx::common
