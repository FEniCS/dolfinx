// Copyright (C) 2003-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "timing.h"
#include "Timer.h"
#include <dolfin/common/Table.h>
#include <dolfin/common/TimeLogManager.h>
#include <tuple>

namespace dolfin
{
namespace common
{
Timer __global_timer;
Timer __tic_timer;
} // namespace common
} // namespace dolfin

using namespace dolfin;
using namespace dolfin::common;

//-----------------------------------------------------------------------
void dolfin::tic() { __tic_timer.start(); }
//-----------------------------------------------------------------------------
double dolfin::toc() { return std::get<0>(__tic_timer.elapsed()); }
//-----------------------------------------------------------------------------
double dolfin::time() { return std::get<0>(__global_timer.elapsed()); }
//-----------------------------------------------------------------------------
Table dolfin::timings(std::set<TimingType> type)
{
  return TimeLogManager::logger().timings(type);
}
//-----------------------------------------------------------------------------
void dolfin::list_timings(std::set<TimingType> type)
{
  TimeLogManager::logger().list_timings(type);
}
//-----------------------------------------------------------------------------
std::tuple<std::size_t, double, double, double> dolfin::timing(std::string task)
{
  return TimeLogManager::logger().timing(task);
}
//-----------------------------------------------------------------------------
