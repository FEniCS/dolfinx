#include <time.h>
#include <iostream>
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

  std::cout << "- Elapsed time: " << elapsed_time << " seconds" << std::endl;

  return elapsed_time;
}
//-----------------------------------------------------------------------------
