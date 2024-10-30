// Copyright (C) 2005-2010 Anders Logg, 2015 Jan Blechta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Table.h"
#include <map>
#include <mpi.h>
#include <string>
#include <utility>

namespace dolfinx
{
/// @brief Return a summary of timings and tasks in a Table.
/// @return Table with timings.
Table timing_table();

/// @brief List a summary of timings and tasks.
///
/// ``MPI_AVG`` reduction is printed.
///
/// @param[in] comm MPI Communicator.
/// @param[in] reduction MPI Reduction to apply (min, max or average).
void list_timings(MPI_Comm comm,
                  Table::Reduction reduction = Table::Reduction::max);

/// @brief Return timing (count, total wall time) for given task.
/// @param[in] task Name of a task
/// @return The (count, total wall time) for the task.
std::pair<int, double> timing(std::string task);

/// @brief Logged elapsed times.
/// @return Elapsed [task id: (count, total wall time)].
std::map<std::string, std::pair<int, double>> timings();

} // namespace dolfinx
