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
TimeLogger::TimeLogger() : _mpi_comm(MPI_COMM_WORLD)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void TimeLogger::register_timing(std::string task,
                                 std::tuple<double, double, double> elapsed)
{
  assert(elapsed >= std::make_tuple(double(0.0), double(0.0), double(0.0)));

  // Print a message
  std::stringstream line;
  line << "Elapsed wall, usr, sys time: " << std::get<0>(elapsed) << ", "
       << std::get<1>(elapsed) << ", " << std::get<2>(elapsed) << " (" << task
       << ")";
  DLOG(INFO) << line.str();

  // Store values for summary
  const auto timing = std::tuple_cat(std::make_tuple(std::size_t(1)), elapsed);
  auto it = _timings.find(task);
  if (it == _timings.end())
  {
    _timings[task] = timing;
  }
  else
  {
    std::get<0>(it->second) += std::get<0>(timing);
    std::get<1>(it->second) += std::get<1>(timing);
    std::get<2>(it->second) += std::get<2>(timing);
    std::get<3>(it->second) += std::get<3>(timing);
  }
}
//-----------------------------------------------------------------------------
void TimeLogger::list_timings(std::set<TimingType> type)
{
  // Format and reduce to rank 0
  Table timings = this->timings(type);
  timings = MPI::avg(_mpi_comm, timings);
  const std::string str = "\n" + timings.str(true);

  // Print just on rank 0
  if (dolfin::MPI::rank(_mpi_comm) == 0)
    std::cout << str << std::endl;
}
//-----------------------------------------------------------------------------
std::map<TimingType, std::string> TimeLogger::_TimingType_descr
    = {{TimingType::wall, "wall"},
       {TimingType::user, "usr"},
       {TimingType::system, "sys"}};
//-----------------------------------------------------------------------------
Table TimeLogger::timings(std::set<TimingType> type)
{
  // Generate log::timing table
  Table table("Summary of timings");
  for (auto& it : _timings)
  {
    const std::string task = it.first;
    const std::size_t num_timings = std::get<0>(it.second);
    const std::vector<double> times{
        std::get<1>(it.second), std::get<2>(it.second), std::get<3>(it.second)};
    table(task, "reps") = num_timings;
    for (const auto& t : type)
    {
      const double total_time = times[static_cast<int>(t)];
      const double average_time = total_time / static_cast<double>(num_timings);
      table(task, TimeLogger::_TimingType_descr[t] + " avg") = average_time;
      table(task, TimeLogger::_TimingType_descr[t] + " tot") = total_time;
    }
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
  {
    std::stringstream line;
    line << "No timings registered for task \"" << task << "\".";
    throw std::runtime_error("Cannot extract timing. " + line.str());
  }
  // Prepare for return for the case of reset
  const auto result = it->second;

  return result;
}
//-----------------------------------------------------------------------------
