// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2010.
//
// First added:  2006-08-18
// Last changed: 2010-03-30

#include <dolfin.h>

using namespace dolfin;

#define NUM_REPS 100
#define SIZE 1000000

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  info("Assigning to vector of size %d (%d repetitions)",
       SIZE, NUM_REPS);

  parameters.parse(argc, argv);

  Vector x(SIZE);

  for (unsigned int i = 0; i < NUM_REPS; i++)
    for (unsigned int j = 0; j < SIZE; j++)
      x.setitem(j, 1.0);

  return 0;
}
//-----------------------------------------------------------------------------
