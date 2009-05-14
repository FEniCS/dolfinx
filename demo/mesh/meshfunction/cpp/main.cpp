// Copyright (C) 2006 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2007.
//
// First added:  2006-11-29
// Last changed: 2008-03-31

#include <dolfin.h>
#include <dolfin/io/NewXMLFile.h>

using namespace dolfin;

int main()
{
  // Read mesh from file (new style)
  File inmesh("../mesh2D.xml.gz", true);
  Mesh mesh;
  inmesh >> mesh;

  // Read mesh function from file (new style)
  File in("../meshfunction.xml", true);
  MeshFunction<double> f(mesh);
  in >> f;

  // Write mesh function to file (new style)
  File out("meshfunction_out.xml", true);
  out << f;

  // Plot mesh function
  plot(f);
}
