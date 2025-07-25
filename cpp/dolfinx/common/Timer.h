// Copyright (C) 2008-2015 Anders Logg, Jan Blechta, Paul T. Kühner and Garth N.
// Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "TimeLogger.h"
#include <chrono>
#include <optional>
#include <stdexcept>
#include <string>

namespace dolfinx::common
{
/// @brief Timer for measuring and logging elapsed time durations.
///
/// The basic usage is
/// \code{.cpp}
/// Timer timer("Assembling over cells");
/// \endcode
/// The timer is started at construction and timing ends when the timer
/// is destroyed (goes out-of-scope). The timer can be started (reset)
/// and stopped explicitly by
/// \code{.cpp}
///   timer.start();
///    /* .... */
///   timer.stop();
/// \endcode
/// A summary of registered elapsed times can be printed by calling:
/// \code{.cpp}
///   list_timings();
/// \endcode
/// Registered elapsed times are logged when (1) the timer goes
/// out-of-scope or (2) Timer::flush() is called.
template <typename T = std::chrono::high_resolution_clock>
class Timer
{
public:
  /// @brief Create and start timer.
  ///
  /// Elapsed time is optionally registered in the logger when the Timer
  /// destructor is called.
  ///
  /// @param[in] task Name used to registered the elapsed time in the
  /// logger. If no name is set, the elapsed time is not registered in
  /// the logger.
  Timer(std::optional<std::string> task = std::nullopt) : _task(std::move(task))
  {
  }

  /// If timer is still running, it is stopped. Elapsed time is
  /// registered in the logger.
  ~Timer()
  {
    if (_start_time.has_value() and _task.has_value())
    {
      _acc += T::now() - *_start_time;
      TimeLogger::instance().register_timing(*_task, _acc);
    }
  }

  /// Reset elapsed time and (re-)start timer.
  void start()
  {
    _acc = T::duration::zero();
    _start_time = T::now();
  }

  /// @brief Elapsed time since time has been started.
  ///
  /// Default duration unit is seconds.
  ///
  /// @return Elapsed time duration.
  template <typename Period = std::ratio<1>>
  std::chrono::duration<double, Period> elapsed() const
  {
    if (_start_time.has_value()) // Timer is running
      return T::now() - *_start_time + _acc;
    else // Timer is stopped
      return _acc;
  }

  /// @brief Stop timer and return elapsed time.
  ///
  /// Default duration unit is seconds.
  ///
  /// @return Elapsed time duration.
  template <typename Period = std::ratio<1>>
  std::chrono::duration<double, Period> stop()
  {
    if (_start_time.has_value()) // Timer is running
    {
      _acc += T::now() - *_start_time;
      _start_time = std::nullopt;
    }

    return _acc;
  }

  /// @brief Resume a stopped timer.
  ///
  /// Does nothing if timer has not been stopped.
  void resume()
  {
    if (!_start_time.has_value())
      _start_time = T::now();
  }

  /// @brief Flush timer duration to the logger.
  ///
  /// An instance of a timer can be flushed to the logger only once.
  /// Subsequent calls will have no effect and will not trigger any
  /// logging.
  ///
  /// @pre Timer must have been stopped before flushing.
  void flush()
  {
    if (_start_time.has_value())
      throw std::runtime_error("Timer must be stopped before flushing.");

    if (_task.has_value())
    {
      TimeLogger::instance().register_timing(*_task, _acc);
      _task = std::nullopt;
    }
  }

private:
  // Name of task to register in logger
  std::optional<std::string> _task;

  // Elapsed time offset
  T::duration _acc = T::duration::zero();

  // Store start time (std::nullopt if timer has been stopped)
  std::optional<typename T::time_point> _start_time = T::now();
};
} // namespace dolfinx::common
