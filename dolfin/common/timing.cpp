// Copyright (C) 2003-2011 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2003-12-21
// Last changed: 2012-11-01

#include <tuple>

#include <dolfin/log/log.h>
#include <dolfin/log/LogManager.h>
#include <dolfin/log/Table.h>
#include "Timer.h"
#include "timing.h"

namespace dolfin
{
  Timer __global_timer;
  Timer __tic_timer;
}

using namespace dolfin;

//-----------------------------------------------------------------------
void dolfin::tic()
{
  __tic_timer.start();
}
//-----------------------------------------------------------------------------
double dolfin::toc()
{
  return std::get<0>(__tic_timer.elapsed());
}
//-----------------------------------------------------------------------------
double dolfin::time()
{
  return std::get<0>(__global_timer.elapsed());
}
//-----------------------------------------------------------------------------
Table dolfin::timings(TimingClear clear, std::set<TimingType> type)
{
  return LogManager::logger().timings(clear, type);
}
//-----------------------------------------------------------------------------
void dolfin::list_timings(TimingClear clear, std::set<TimingType> type)
{
  LogManager::logger().list_timings(clear, type);
}
//-----------------------------------------------------------------------------
void dolfin::dump_timings_to_xml(std::string filename, TimingClear clear)
{
  LogManager::logger().dump_timings_to_xml(filename, clear);
}
//-----------------------------------------------------------------------------
std::tuple<std::size_t, double, double, double>
  dolfin::timing(std::string task, TimingClear clear)
{
  return LogManager::logger().timing(task, clear);
}
//-----------------------------------------------------------------------------
