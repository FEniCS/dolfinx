// Copyright (C) 2003-2016 Anders Logg
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "TimeLogger.h"
#include "MPI.h"
#include "log.h"
#include <iostream>

using namespace dolfinx;
using namespace dolfinx::common;

//-----------------------------------------------------------------------------
void TimeLogger::register_timing(std::string task, double time)
{
  assert(time >= 0.0);

  // Print a message
  std::string line
      = "Elapsed time: " + std::to_string(time) + " (" + task + ")";
  spdlog::debug(line.c_str());

  // Store values for summary
  if (auto it = _timings.find(task); it != _timings.end())
  {
    std::get<0>(it->second) += 1;
    std::get<1>(it->second) += time;
  }
  else
    _timings.insert({task, {1, time}});
}
//-----------------------------------------------------------------------------
void TimeLogger::list_timings(MPI_Comm comm, Table::Reduction reduction) const
{
  // Format and reduce to rank 0
  Table timings = this->timings();
  timings = timings.reduce(comm, reduction);
  const std::string str = "\n" + timings.str();

  // Print just on rank 0
  if (dolfinx::MPI::rank(comm) == 0)
    std::cout << str << std::endl;
}
//-----------------------------------------------------------------------------
Table TimeLogger::timings() const
{
  // Generate log::timing table
  Table table("Summary of timings");
  for (auto& it : _timings)
  {
    std::string task = it.first;
    auto [num_timings, time] = it.second;
    table.set(task, "reps", num_timings);
    table.set(task, "avg", time / static_cast<double>(num_timings));
    table.set(task, "tot", time);
  }

  return table;
}
//-----------------------------------------------------------------------------
std::pair<int, double> TimeLogger::timing(std::string task) const
{
  // Find timing
  auto it = _timings.find(task);
  if (it == _timings.end())
  {
    throw std::runtime_error("No timings registered for task \"" + task
                             + "\".");
  }

  return it->second;
}
//-----------------------------------------------------------------------------
