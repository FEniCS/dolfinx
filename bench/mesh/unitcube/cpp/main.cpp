// Copyright (C) 2006-2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-11-01
// Last changed: 2010-05-03

#include <dolfin.h>

using namespace dolfin;

#define NUM_REPS 10
#define SIZE 128

int main(int argc, char* argv[])
{
  info("Creating unit cube of size %d x %d x %d (%d repetitions)",
       SIZE, SIZE, SIZE, NUM_REPS);

  parameters.parse(argc, argv);

  for (int i = 0; i < NUM_REPS; i++)
  {
    UnitCube mesh(SIZE, SIZE, SIZE);
    dolfin::cout << "Created unit cube: " << mesh << dolfin::endl;
  }

  return 0;
}
