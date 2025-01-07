// Copyright (C) 2003-2011 Anders Logg
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "timing.h"
#include "Table.h"
#include "TimeLogger.h"
#include "Timer.h"

//-----------------------------------------------------------------------
dolfinx::Table dolfinx::timing_table()
{
  return dolfinx::common::TimeLogger::instance().timing_table();
}
//-----------------------------------------------------------------------------
void dolfinx::list_timings(MPI_Comm comm, Table::Reduction reduction)
{
  dolfinx::common::TimeLogger::instance().list_timings(comm, reduction);
}
//-----------------------------------------------------------------------------
std::pair<int, std::chrono::duration<double, std::ratio<1>>>
dolfinx::timing(std::string task)
{
  return dolfinx::common::TimeLogger::instance().timing(task);
}
//-----------------------------------------------------------------------------
std::map<std::string,
         std::pair<int, std::chrono::duration<double, std::ratio<1>>>>
dolfinx::timings()
{
  return dolfinx::common::TimeLogger::instance().timings();
}
//-----------------------------------------------------------------------------
