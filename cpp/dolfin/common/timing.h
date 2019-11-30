// Copyright (C) 2005-2010 Anders Logg, 2015 Jan Blechta
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <dolfin/common/Table.h>
#include <mpi.h>
#include <set>
#include <string>
#include <tuple>

namespace dolfin
{

/// Timing types:
///   * ``TimingType::wall`` wall-clock time
///   * ``TimingType::user`` user (cpu) time
///   * ``TimingType::system`` system (kernel) time
enum class TimingType : std::int32_t
{
  wall = 0,
  user = 1,
  system = 2
};

/// Return a summary of timings and tasks in a Table
/// @param[in] type subset of { TimingType::wall, TimingType::user,
///                 TimingType::system }
/// @returns Table with timings
Table timings(std::set<TimingType> type);

/// List a summary of timings and tasks. ``MPI_AVG`` reduction is
/// printed.
/// @param[in] mpi_comm MPI Communicator
/// @param[in] type Subset of { TimingType::wall, TimingType::user,
///                 TimingType::system }
void list_timings(MPI_Comm mpi_comm, std::set<TimingType> type);

/// Return timing (count, total wall time, total user time, total system
/// time) for given task.
/// @param[in] task Name of a task
/// @returns The (count, total wall time, total user time, total system
///          time) for the task
std::tuple<std::size_t, double, double, double> timing(std::string task);

} // namespace dolfin
