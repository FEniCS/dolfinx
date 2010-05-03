// Copyright (C) 2003-2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-12-21
// Last changed: 2010-05-03

#include <ctime>
#include <dolfin/log/dolfin_log.h>
#include "timing.h"

namespace dolfin
{
  clock_t __tic_time;
}

using namespace dolfin;

//-----------------------------------------------------------------------------
void dolfin::tic()
{
  dolfin::__tic_time = std::clock();
}
//-----------------------------------------------------------------------------
double dolfin::toc()
{
  clock_t __toc_time = std::clock();

  double elapsed_time = ((double) (__toc_time - __tic_time)) / CLOCKS_PER_SEC;
  if (elapsed_time < 1e-10)
    elapsed_time = 0;

  return elapsed_time;
}
//-----------------------------------------------------------------------------
double dolfin::time()
{
  clock_t __toc_time = std::clock();
  return ((double) (__toc_time)) / CLOCKS_PER_SEC;
}
//-----------------------------------------------------------------------------
