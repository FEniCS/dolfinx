// Copyright (C) 2007 Anders Logg and Magnus Vikström.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-05-08
// Last changed: 2007-12-18

#include <dolfin.h>

using namespace dolfin;

int main()
{
  // Create mesh
  UnitCube mesh(16, 16, 16);

  // Partition mesh
  MeshFunction<unsigned int> partitions;
  mesh.partition(partitions, 20);

  // Display partition
  partitions.disp();

  // Plot mesh partition
  plot(partitions);

}
