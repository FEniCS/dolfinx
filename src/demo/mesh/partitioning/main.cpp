// Copyright (C) 2007 Anders Logg and Magnus Vikström.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-05-08
// Last changed: 2007-05-08

#include <dolfin.h>

using namespace dolfin;

int main()
{
  // Create mesh
  UnitSquare mesh(8, 8);

  // FIXME: Should work just like this, need to call partitions.init(...)
  // MeshFunction<unsigned int> partitions;

  // Partition mesh
  MeshFunction<unsigned int> partitions(mesh, 2);
  mesh.partition(4, partitions);

  // Display partitions
  partitions.disp();

  // Plot mesh partition
  plot(partitions);

  return 0;
}
