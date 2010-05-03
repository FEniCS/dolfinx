// Copyright (C) 2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2010.
//
// First added:  2010-03-30
// Last changed: 2010-05-03

#include <dolfin.h>

using namespace dolfin;

#define SIZE 10000000
#define NUM_REPS 100

int main(int argc, char* argv[])
{
  info("Accessing vector of size %d (%d repetitions)",
       SIZE, NUM_REPS);

  parameters.parse(argc, argv);

  Vector x(SIZE);
  x.zero();

  double sum = 0.0;
  for (unsigned int i = 0; i < NUM_REPS; i++)
    for (unsigned int j = 0; j < SIZE; j++)
      sum += x[j];
  dolfin::cout << "Sum is " << sum << dolfin::endl;

  return 0;
}
