// Copyright (C) 2008 Anders Logg and Magnus Vikström.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-05-08
// Last changed: 2008-04-11

#include <dolfin.h>

using namespace dolfin;

int main()
{
#ifndef HAS_SCOTCH
  message("Sorry, this demo requires SCOTCH.");
  return 0;
#endif

  // Create mesh
  UnitCube mesh(16, 16, 16);

  // Partition mesh
  MeshFunction<unsigned int> partitions;
  mesh.partition(partitions, 20);

  // Plot mesh partition
  plot(partitions);

  return 0;
}
