// Copyright (C) 2003-2011 Anders Logg
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "timing.h"
#include "Table.h"
#include "TimeLogManager.h"
#include "TimeLogger.h"
#include "Timer.h"

//-----------------------------------------------------------------------
dolfinx::Table dolfinx::timing_table()
{
  return dolfinx::common::TimeLogManager::logger().timing_table();
}
//-----------------------------------------------------------------------------
void dolfinx::list_timings(MPI_Comm comm, Table::Reduction reduction)
{
  dolfinx::common::TimeLogManager::logger().list_timings(comm, reduction);
}
//-----------------------------------------------------------------------------
std::pair<int, double> dolfinx::timing(std::string task)
{
  return dolfinx::common::TimeLogManager::logger().timing(task);
}
//-----------------------------------------------------------------------------
std::map<std::string, std::pair<int, double>> dolfinx::timings()
{
  return dolfinx::common::TimeLogManager::logger().timings();
}
//-----------------------------------------------------------------------------
