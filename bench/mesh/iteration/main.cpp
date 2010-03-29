// Copyright (C) 2006-2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-11-01
// Last changed: 2010-03-30

#include <dolfin.h>

using namespace dolfin;

#define NUM_REPS 1000
#define MESH_SIZE 32

int main()
{
  info("Iteration over mesh entities, %d repetitions on a unit cube of size %d x %d x %d",
       NUM_REPS, MESH_SIZE, MESH_SIZE, MESH_SIZE);

  UnitCube mesh(MESH_SIZE, MESH_SIZE, MESH_SIZE);

  int sum = 0;
  for (int i = 0; i < NUM_REPS; i++)
  {
    for (CellIterator c(mesh); !c.end(); ++c)
      for (VertexIterator v(*c); !v.end(); ++v)
        sum += v->index();
  }
  info("Sum is %d", sum);

  return 0;
}
