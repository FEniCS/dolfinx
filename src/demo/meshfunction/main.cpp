// Copyright (C) 2006 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-11-29
// Last changed: 2006-11-30

#include <dolfin.h>

using namespace dolfin;

int main()
{
  Mesh mesh("mesh2D.xml.gz");

  File in("meshfunction.xml");
  MeshFunction<double> f(mesh);
  in >> f;

  File out("meshfunction_out.xml");
  out << f;
} 
