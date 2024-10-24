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
dolfinx::Table dolfinx::timings()
{
  return dolfinx::common::TimeLogManager::logger().timings();
}
//-----------------------------------------------------------------------------
void dolfinx::list_timings(MPI_Comm comm, Table::Reduction reduction)
{
  dolfinx::common::TimeLogManager::logger().list_timings(comm, reduction);
}
//-----------------------------------------------------------------------------
std::tuple<std::size_t, double>
dolfinx::timing(std::string task)
{
  return dolfinx::common::TimeLogManager::logger().timing(task);
}
//-----------------------------------------------------------------------------
