// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-10-26
// Last changed: 2006-10-26

#include <dolfin.h>

using namespace dolfin;

int main()
{
  Mesh mesh("mesh2D.xml.gz");
  //Mesh mesh("mesh3D.xml.gz");

  mesh.refine();
  
  cout << "Iterating over the cells in the mesh..." << endl;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    cout << *cell << endl;
  
  BoundaryMesh boundary(mesh);  

  cout << "Iterating over the cells in the boundary..." << endl;
  for (CellIterator facet(boundary); !facet.end(); ++facet)
    cout << *facet << endl;

  return 0;
}
