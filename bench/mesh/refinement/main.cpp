// Copyright (C) 2006-2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-11-01
// Last changed: 2010-03-30

#include <dolfin.h>

using namespace dolfin;

#define NUM_REPS 5
#define MESH_SIZE 4

int main(int argc, char* argv[])
{
  info("Uniform mesh refinement, %d refinements of a unit cube of size %d x %d x %d",
       NUM_REPS, MESH_SIZE, MESH_SIZE, MESH_SIZE);

  parameters.parse(argc, argv);

  UnitCube mesh(MESH_SIZE, MESH_SIZE, MESH_SIZE);

  for (int i = 0; i < NUM_REPS; i++)
  {
    mesh.refine();
    dolfin::cout << "Refined mesh: " << mesh << dolfin::endl;
  }

  return 0;
}
