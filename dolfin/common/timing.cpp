// Copyright (C) 2003-2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-12-21
// Last changed: 2010-11-16

// Uncomment this for testing std::clock
#define _WIN32

#ifdef _WIN32
#include <boost/timer.hpp>
#else
#include <sys/time.h>
#endif

#include <dolfin/log/log.h>
#include "timing.h"

// We use boost::timer (std::clock) on Windows and otherwise the
// platform-dependent (but higher-precision) gettimeofday from
// <sys/time.h>. Note that in the latter case, the timer is not
// reset to zero at the start of the program so time() will not
// report total CPU time, only the difference makes sense.

namespace dolfin
{
#ifdef _WIN32
  boost::timer __global_timer;
  boost::timer __tic_timer;
#else
  double __tic_timer;
#endif
}

using namespace dolfin;

//-----------------------------------------------------------------------
void dolfin::tic()
{
#ifdef _WIN32
  dolfin::__tic_timer.restart();
#else
  dolfin::__tic_timer = time();
#endif
}
//-----------------------------------------------------------------------------
double dolfin::toc()
{
#ifdef _WIN32
  return __tic_timer.elapsed();
#else
  return time() - __tic_timer;
#endif
}
//-----------------------------------------------------------------------------
double dolfin::time()
{
#ifdef _WIN32
  return dolfin::__global_timer.elapsed();
#else
  struct timeval tv;
  struct timezone tz;
  if (gettimeofday(&tv, &tz) != 0)
    error("Timing failed, gettimeofday() failed.");
  return static_cast<double>(tv.tv_sec) + static_cast<double>(tv.tv_usec)*1e-6;
#endif
}
//-----------------------------------------------------------------------------
