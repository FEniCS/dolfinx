// Copyright (C) 2003-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "timing.h"
#include "Timer.h"
#include <dolfin/common/Table.h>
#include <dolfin/common/TimeLogManager.h>

namespace
{
dolfin::common::Timer __global_timer;
dolfin::common::Timer __tic_timer;
} // namespace

using namespace dolfin;
using namespace dolfin::common;

//-----------------------------------------------------------------------
Table dolfin::timings(std::set<TimingType> type)
{
  return TimeLogManager::logger().timings(type);
}
//-----------------------------------------------------------------------------
void dolfin::list_timings(MPI_Comm mpi_comm, std::set<TimingType> type)
{
  TimeLogManager::logger().list_timings(mpi_comm, type);
}
//-----------------------------------------------------------------------------
std::tuple<std::size_t, double, double, double> dolfin::timing(std::string task)
{
  return TimeLogManager::logger().timing(task);
}
//-----------------------------------------------------------------------------
