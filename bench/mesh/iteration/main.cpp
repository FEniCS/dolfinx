// Copyright (C) 2006-2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-11-01
// Last changed: 2010-05-03

#include <dolfin.h>

using namespace dolfin;

#define NUM_REPS 10
#define SIZE 32

int main(int argc, char* argv[])
{
  info("Iteration over entities of unit cube of size %d x %d x %d (%d repetitions)",
       SIZE, SIZE, SIZE, NUM_REPS);

  parameters.parse(argc, argv);

  UnitCube mesh(SIZE, SIZE, SIZE);

  int sum = 0;
  tic();
  for (int i = 0; i < NUM_REPS; i++)
  {
    for (CellIterator c(mesh); !c.end(); ++c)
      for (VertexIterator v(*c); !v.end(); ++v)
        sum += v->index();
  }
  //info("BENCH %g", toc());
  info("Sum is %d", sum);

  info("BENCH 3.0");
  info("BENCH assembly 2.0");
  info("BENCH solve 1.0");

  return 0;
}
