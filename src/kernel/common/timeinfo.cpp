// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <time.h>

#include <dolfin/dolfin_log.h>
#include <dolfin/timeinfo.h>

using namespace dolfin;

namespace dolfin {

  clock_t __tic_time;

}

//-----------------------------------------------------------------------------
void dolfin::tic()
{
  dolfin::__tic_time = clock();
}
//-----------------------------------------------------------------------------
real dolfin::toc()
{
  clock_t __toc_time = clock();

  double elapsed_time = ((double) (__toc_time - __tic_time)) / CLOCKS_PER_SEC;

  cout << "Elapsed time: " << elapsed_time << " seconds" << endl;

  return elapsed_time;
}
//-----------------------------------------------------------------------------
