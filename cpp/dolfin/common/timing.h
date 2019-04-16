// Copyright (C) 2005-2010 Anders Logg, 2015 Jan Blechta
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <dolfin/common/Table.h>
#include <string>

namespace dolfin
{

/// Timing types:
///   * ``TimingType::wall`` wall-clock time
///   * ``TimingType::user`` user (cpu) time
///   * ``TimingType::system`` system (kernel) time
///
/// Precision of wall is around 1 microsecond, user and system are around
/// 10 millisecond (on Linux).
enum class TimingType : int32_t
{
  wall = 0,
  user = 1,
  system = 2
};

/// Start timing (should not be used internally in DOLFIN!)
void tic();

/// Return elapsed wall time (should not be used internally in DOLFIN!)
double toc();

/// Return wall time elapsed since some implementation dependent epoch
double time();

/// Return a summary of timings and tasks in a _Table_
///
/// @param    type (std::set<TimingType>)
///         subset of ``{ TimingType::wall, TimingType::user,
///         TimingType::system }``
///
/// @returns    _Table_
///         _Table_ with timings
Table timings(std::set<TimingType> type);

/// List a summary of timings and tasks.
/// ``MPI_AVG`` reduction is printed. Collective on ``MPI_COMM_WORLD``.
///
/// @param    type (std::set<TimingType>)
///         subset of ``{ TimingType::wall, TimingType::user,
///         TimingType::system }``
void list_timings(std::set<TimingType> type);
// NOTE: Function marked as 'collective on COMM_WORLD' (instead of
//       'collective on Logger::mpi_comm()') as user has no clue what the
//       function has to do with Logger

/// Dump a summary of timings and tasks to XML file.
/// ``MPI_MAX``, ``MPI_MIN`` and ``MPI_AVG`` reductions are
/// stored. Collective on ``MPI_COMM_WORLD``.
///
/// @param    filename (std::string)
///         output filename; must have ``.xml`` suffix; existing file
///         is silently overwritten
void dump_timings_to_xml(std::string filename);
// NOTE: Function marked as 'collective on COMM_WORLD' (instead of
//       'collective on Logger::mpi_comm()') as user has no clue what the
//       function has to do with Logger

/// Return timing (count, total wall time, total user time,
/// total system time) for given task.
///
/// @param    task (std::string)
///         name of a task
///
/// @returns    std::tuple<std::size_t, double, double, double>
///         (count, total wall time, total user time, total system time)
std::tuple<std::size_t, double, double, double> timing(std::string task);

} // namespace dolfin
