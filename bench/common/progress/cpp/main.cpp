// Copyright (C) 2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-06-29
// Last changed: 2010-11-16

#include <dolfin.h>

using namespace dolfin;

#define NUM_REPS 5
#define SIZE 500000000

int main(int argc, char* argv[])
{
  info("Creating progress bar with %d steps (%d repetitions)",
       SIZE, NUM_REPS);

  for (int i = 0; i < NUM_REPS; i++)
  {
    Progress p("Stepping", SIZE);
    double sum = 0.0;
    for (int j = 0; j < SIZE; j++)
    {
      sum += 0.1;
      p++;
    }
    dolfin::cout << "sum = " << sum << dolfin::endl;
  }

  return 0;
}
