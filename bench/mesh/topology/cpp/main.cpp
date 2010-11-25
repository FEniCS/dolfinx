// Copyright (C) 2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-11-25
// Last changed: 2010-11-25

#include <dolfin.h>

using namespace dolfin;

#define NUM_REPS 5
#define SIZE 64

// Use for quick testing
//#define NUM_REPS 2
//#define SIZE 32

int main(int argc, char* argv[])
{
  info("Creating cell-cell connectivity for unit cube of size %d x %d x %d (%d repetitions)",
       SIZE, SIZE, SIZE, NUM_REPS);

  parameters.parse(argc, argv);

  UnitCube mesh(SIZE, SIZE, SIZE);
  const int D = mesh.topology().dim();

  for (int i = 0; i < NUM_REPS; i++)
  {
    mesh.clean();
    mesh.init(D, D);
    dolfin::cout << "Created unit cube: " << mesh << dolfin::endl;
  }

  summary();

  return 0;
}
