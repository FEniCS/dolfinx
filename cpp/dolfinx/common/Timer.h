// Copyright (C) 2008 Anders Logg, 2015 Jan Blechta, 2024 Paul T. Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "TimeLogManager.h"
#include <array>
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
/// Timings are stored globally and a summary may be printed by calling
///
///   list_timings();
template <typename T = std::chrono::high_resolution_clock>
class Timer
{
public:
  /// @brief Create timer.
  ///
  /// If a task name is provided this enables logging to logger,
  /// otherwise (i.e. no task provided) nothing gets logged.
  Timer(std::optional<std::string> task = std::nullopt) : _task(task) {}

  /// Destructor
  ~Timer() = default;

  /// Zero and start timer
  void start() { _start_time = T::now(); }

  /// @brief Returns elapsed time since time has been started.
  /// @tparam unit to which the time difference is cast
  auto elapsed() const { return T::now() - _start_time; }

  /// Stop timer and return elapsed (wall) time. Also registers timing
  /// data into the logger.
  auto stop()
  {
    auto elapsed = this->elapsed();
    if (_task.has_value())
      TimeLogManager::logger().register_timing(_task.value(), elapsed.count());
    return elapsed;
  }

private:
  // Name of task
  std::optional<std::string> _task;

  T::time_point _start_time;
};
} // namespace dolfinx::common
