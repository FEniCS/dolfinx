// Copyright (C) 2006 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2007.
//
// First added:  2006-11-29
// Last changed: 2008-03-31

#include <dolfin.h>

using namespace dolfin;

int main()
{
  dolfin_set("debug level", 1);

  // Read mesh from file
  Mesh mesh("../mesh2D.xml.gz");
  
  // Read mesh function from file
  File in("../meshfunction.xml");
  MeshFunction<double> f(mesh);
  in >> f;

  // Write mesh function to file
  File out("meshfunction_out.xml");
  out << f;

  // Plot mesh function
  plot(f);
} 
