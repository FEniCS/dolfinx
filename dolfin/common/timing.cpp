// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-12-21
// Last changed: 2005-12-21

#include <ctime>
#include <dolfin/log/dolfin_log.h>
#include "timing.h"

#include "utils.h"

namespace dolfin
{
  clock_t __tic_time;
}

using namespace dolfin;

//-----------------------------------------------------------------------------
void dolfin::tic()
{
  dolfin::__tic_time = clock();
}
//-----------------------------------------------------------------------------
real dolfin::toc()
{
  clock_t __toc_time = clock();

  real elapsed_time = ((real) (__toc_time - __tic_time)) / CLOCKS_PER_SEC;

  return elapsed_time;
}
//-----------------------------------------------------------------------------
real dolfin::tocd()
{
  real elapsed_time = toc();
  
  cout << "Current date: " << date() << endl;

  cout << "Elapsed time: " << elapsed_time << " seconds" << endl;

  return elapsed_time;
}
//-----------------------------------------------------------------------------
