// Copyright (C) 2003-2016 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "TimeLogger.h"
#include <dolfin/common/MPI.h>
#include <dolfin/common/log.h>
#include <vector>

using namespace dolfin;
using namespace dolfin::common;

//-----------------------------------------------------------------------------
TimeLogger::TimeLogger()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void TimeLogger::register_timing(std::string task, double wall, double user,
                                 double system)
{
  assert(wall >= 0.0);
  assert(user >= 0.0);
  assert(system >= 0.0);

  // Print a message
  std::stringstream line;
  line << "Elapsed wall, usr, sys time: " << wall << ", " << user << ", "
       << system << " (" << task << ")";
  DLOG(INFO) << line.str();

  // Store values for summary
  if (auto it = _timings.find(task); it != _timings.end())
  {
    std::get<0>(it->second) += 1;
    std::get<1>(it->second) += wall;
    std::get<2>(it->second) += user;
    std::get<3>(it->second) += system;
  }
  else
    _timings.insert({task, {std::size_t(1), wall, user, system}});
}
//-----------------------------------------------------------------------------
void TimeLogger::list_timings(MPI_Comm mpi_comm, std::set<TimingType> type)
{
  // Format and reduce to rank 0
  Table timings = this->timings(type);
  timings = timings.reduce(mpi_comm, Table::Reduction::average);
  const std::string str = "\n" + timings.str(true);

  // Print just on rank 0
  if (dolfin::MPI::rank(mpi_comm) == 0)
    std::cout << str << std::endl;
}
//-----------------------------------------------------------------------------
Table TimeLogger::timings(std::set<TimingType> type)
{
  // Generate log::timing table
  Table table("Summary of timings");
  for (auto& it : _timings)
  {
    const std::string task = it.first;
    const auto [num_timings, wall, usr, sys] = it.second;
    table(task, "reps") = num_timings;
    table(task, "wall avg") = wall / static_cast<double>(num_timings);
    table(task, "wall tot") = wall;
    table(task, "usr avg") = usr / static_cast<double>(num_timings);
    table(task, "usr tot") = usr;
    table(task, "sys avg") = sys / static_cast<double>(num_timings);
    table(task, "sys tot") = sys;
  }

  return table;
}
//-----------------------------------------------------------------------------
std::tuple<std::size_t, double, double, double>
TimeLogger::timing(std::string task)
{
  // Find timing
  auto it = _timings.find(task);
  if (it == _timings.end())
    throw std::runtime_error("No timings registered for task \"" + task + "\".");

  return it->second;
}
//-----------------------------------------------------------------------------
