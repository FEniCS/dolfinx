// Copyright (C) 2005-2010 Anders Logg, 2015 Jan Blechta
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

#ifndef __TIMING_H
#define __TIMING_H

#include <string>
#include <cstdint>
#include <dolfin/log/Table.h>

/// This comment in in timing.h but I think it is providing
/// a doxygen docstring for the whole dolfin namespace... FIXME.

namespace dolfin
{
  /// Parameter specifying whether to clear timing(s):
  ///   * ``TimingClear::keep``
  ///   * ``TimingClear::clear``
  enum class TimingClear : bool { keep = false, clear = true };

  /// Timing types:
  ///   * ``TimingType::wall`` wall-clock time
  ///   * ``TimingType::user`` user (cpu) time
  ///   * ``TimingType::system`` system (kernel) time
  ///
  /// Precision of wall is around 1 microsecond, user and system are around
  /// 10 millisecond (on Linux).
  enum class TimingType : int32_t { wall = 0, user = 1, system = 2 };

  /// Start timing (should not be used internally in DOLFIN!)
  void tic();

  /// Return elapsed wall time (should not be used internally in DOLFIN!)
  double toc();

  /// Return wall time elapsed since some implementation dependent epoch
  double time();

  /// Return a summary of timings and tasks in a _Table_, optionally clearing
  /// stored timings
  ///
  /// *Arguments*
  ///     clear (TimingClear)
  ///         * ``TimingClear::clear`` resets stored timings
  ///         * ``TimingClear::keep`` leaves stored timings intact
  ///     type (std::set<TimingType>)
  ///         subset of ``{ TimingType::wall, TimingType::user,
  ///         TimingType::system }``
  ///
  /// *Returns*
  ///     _Table_
  ///         _Table_ with timings
  Table timings(TimingClear clear, std::set<TimingType> type);

  /// List a summary of timings and tasks, optionally clearing stored timings.
  /// ``MPI_AVG`` reduction is printed. Collective on ``MPI_COMM_WORLD``.
  ///
  /// *Arguments*
  ///     clear (TimingClear)
  ///         * ``TimingClear::clear`` resets stored timings
  ///         * ``TimingClear::keep`` leaves stored timings intact
  ///     type (std::set<TimingType>)
  ///         subset of ``{ TimingType::wall, TimingType::user,
  ///         TimingType::system }``
  void list_timings(TimingClear clear, std::set<TimingType> type);
  // NOTE: Function marked as 'collective on COMM_WORLD' (instead of
  //       'collective on Logger::mpi_comm()') as user has no clue what the
  //       function has to do with Logger

  /// Dump a summary of timings and tasks to XML file, optionally clearing
  /// stored timings. ``MPI_MAX``, ``MPI_MIN`` and ``MPI_AVG`` reductions are
  /// stored. Collective on ``MPI_COMM_WORLD``.
  ///
  /// *Arguments*
  ///     filename (std::string)
  ///         output filename; must have ``.xml`` suffix; existing file
  ///         is silently overwritten
  ///     clear (TimingClear)
  ///         * ``TimingClear::clear`` resets stored timings
  ///         * ``TimingClear::keep`` leaves stored timings intact
  void dump_timings_to_xml(std::string filename, TimingClear clear);
  // NOTE: Function marked as 'collective on COMM_WORLD' (instead of
  //       'collective on Logger::mpi_comm()') as user has no clue what the
  //       function has to do with Logger

  /// Return timing (count, total wall time, total user time,
  /// total system time) for given task, optionally clearing
  /// all timings for the task
  ///
  /// *Arguments*
  ///     task (std::string)
  ///         name of a task
  ///     clear (TimingClear)
  ///         * ``TimingClear::clear`` resets stored timings
  ///         * ``TimingClear::keep`` leaves stored timings intact
  ///
  /// *Returns*
  ///     std::tuple<std::size_t, double, double, double>
  ///         (count, total wall time, total user time, total system time)
  std::tuple<std::size_t, double, double, double>
    timing(std::string task, TimingClear clear);

}

#endif
