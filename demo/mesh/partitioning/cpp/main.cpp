// Copyright (C) 2008 Ola Skavhaug and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-12-18
// Last changed: 2008-12-18
//
// Run this demo in parallel by
//
//     mpirun -n <n> ./demo
//
// where <n> is the desired number of processes.
// Then plot the partitions by
//
//     ./plotpartitions <n>

#include <dolfin.h>

using namespace dolfin;

int main()
{
  // Read in mesh from XML file in parallel
  Mesh mesh("unitsquare.xml.gz");

  // Store partition to file
  char filename[100];
  sprintf(filename, "unitsquare-%d.xml", dolfin::MPI::process_number());
  File file(filename);
  file << mesh;

  return 0;
}
