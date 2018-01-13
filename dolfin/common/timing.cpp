// Copyright (C) 2003-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <tuple>

#include "Timer.h"
#include "timing.h"
#include <dolfin/log/LogManager.h>
#include <dolfin/log/Table.h>
#include <dolfin/log/log.h>

namespace dolfin
{
Timer __global_timer;
Timer __tic_timer;
}

using namespace dolfin;

//-----------------------------------------------------------------------
void dolfin::tic() { __tic_timer.start(); }
//-----------------------------------------------------------------------------
double dolfin::toc() { return std::get<0>(__tic_timer.elapsed()); }
//-----------------------------------------------------------------------------
double dolfin::time() { return std::get<0>(__global_timer.elapsed()); }
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
std::tuple<std::size_t, double, double, double>
dolfin::timing(std::string task, TimingClear clear)
{
  return LogManager::logger().timing(task, clear);
}
//-----------------------------------------------------------------------------
