// Copyright (C) 2003-2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-12-21
// Last changed: 2010-11-10

#include <boost/timer.hpp>
#include "timing.h"

namespace dolfin
{
  boost::timer __global_timer;
  boost::timer __tic_timer;
}

using namespace dolfin;

//-----------------------------------------------------------------------------
void dolfin::tic()
{
  dolfin::__tic_timer.restart();
}
//-----------------------------------------------------------------------------
double dolfin::toc()
{
  return __tic_timer.elapsed();
}
//-----------------------------------------------------------------------------
double dolfin::time()
{
  return dolfin::__global_timer.elapsed();
}
//-----------------------------------------------------------------------------
