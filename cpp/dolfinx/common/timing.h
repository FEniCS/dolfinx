// Copyright (C) 2005-2010 Anders Logg, 2015 Jan Blechta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Table.h"
#include <mpi.h>
#include <set>
#include <string>
#include <tuple>

namespace dolfinx
{
/// @brief Timing types.
enum class TimingType : int
{
  wall = 0,  ///< Wall-clock time
  user = 1,  ///< User (cpu) time
  system = 2 ///< System (kernel) time
};

/// @brief Return a summary of timings and tasks in a Table.
/// @param[in] type Timing type.
/// @return Table with timings.
Table timings(std::set<TimingType> type);

/// @brief List a summary of timings and tasks.
///
/// ``MPI_AVG`` reduction is printed.
///
/// @param[in] comm MPI Communicator.
/// @param[in] type Timing type.
/// @param[in] reduction MPI Reduction to apply (min, max or average).
void list_timings(MPI_Comm comm, std::set<TimingType> type,
                  Table::Reduction reduction = Table::Reduction::max);

/// @brief Return timing (count, total wall time, total user time, total
/// system time) for given task.
/// @param[in] task Name of a task
/// @return The (count, total wall time, total user time, total system
/// time) for the task.
std::tuple<std::size_t, double, double, double> timing(std::string task);

} // namespace dolfinx
