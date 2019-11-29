// Copyright (C) 2005-2010 Anders Logg, 2015 Jan Blechta
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <mpi.h>
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
/// ``MPI_AVG`` reduction is printed.
///
/// @param mpi_comm MPI Communicator
/// @param  type (std::set<TimingType>)
///         subset of ``{ TimingType::wall, TimingType::user,
///         TimingType::system }``
void list_timings(MPI_Comm mpi_comm, std::set<TimingType> type);

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
