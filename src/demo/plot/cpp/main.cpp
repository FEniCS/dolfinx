// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-05-29
// Last changed: 2007-05-30

#include <dolfin.h>

using namespace dolfin;

int main()
{
  // Read and plot mesh from file
  Mesh mesh("dolfin-2.xml.gz");
  plot(mesh);
}
