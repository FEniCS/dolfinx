// Copyright (C) 2006-2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-11-01
// Last changed: 2010-04-05

#include <dolfin.h>

using namespace dolfin;

#define NUM_REPS 3
#define SIZE 4

int main(int argc, char* argv[])
{
  info("Uniform refinement of unit cube of size %d x %d x %d (%d refinements)",
       SIZE, SIZE, SIZE, NUM_REPS);

  parameters.parse(argc, argv);

  UnitCube mesh(SIZE, SIZE, SIZE);

  for (int i = 0; i < NUM_REPS; i++)
  {
    mesh = refine(mesh);
    dolfin::cout << "Refined mesh: " << mesh << dolfin::endl;
  }

  return 0;
}
