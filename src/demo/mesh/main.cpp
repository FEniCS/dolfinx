// Copyright (C) 2003-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2005.
//
// First added:  2003-10-21
// Last changed: 2005-12-08

#include <dolfin.h>

using namespace dolfin;

void refine2D(int refinements);
void refine3D(int refinements);

int main()
{
  int refinements = 3;

  // Refine 2D mesh
  refine2D(refinements);

  // Refine 3D mesh
  refine3D(refinements);
  
  return 0;
}

void refine2D(int refinements)
{
  dolfin::cout << "Refining 2D mesh" << dolfin::endl;
  dolfin::cout << "----------------" << dolfin::endl;  

  // Load mesh
  Mesh mesh("mesh2D.xml.gz");
  
  // Refine a couple of times
  for (int i = 0; i < refinements; i++)
  {
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      // Mark cells close to y = x
      for (VertexIterator vertex(cell); !vertex.end(); ++vertex)
        if ( fabs(vertex->coord().x - vertex->coord().y) < DOLFIN_EPS )
	         cell->mark();
      
      // Mark cells at the corners
      if ( cell->midpoint().dist(0.0, 0.0) < 0.25 ||
	   cell->midpoint().dist(1.0, 0.0) < 0.25 ||
	   cell->midpoint().dist(1.0, 1.0) < 0.25 ||
	   cell->midpoint().dist(0.0, 1.0) < 0.25 )
      cell->mark();
    }

    // Refine mesh
    mesh.refine();
  }

  // Save all meshes in the mesh hierarchy in VTK format
  File file("mesh2D.pvd");
  MeshHierarchy meshes(mesh);
  for (MeshIterator m(meshes); !m.end(); ++m)
    file << *m;

  dolfin::cout << dolfin::endl;
}

void refine3D(int refinements)
{
  dolfin::cout << "Refining 3D mesh" << dolfin::endl;
  dolfin::cout << "----------------" << dolfin::endl;  

  // Load mesh
  Mesh mesh("mesh3D.xml.gz");
  
  // Refine a couple of times
  for (int i = 0; i < refinements; i++)
  {
    // Mark cells close to (0,0,0)
    for (CellIterator cell(mesh); !cell.end(); ++cell)
      if ( cell->midpoint().dist(0.0, 0.0, 0.0) < 0.3 )
	cell->mark();
    
    // Refine mesh
    mesh.refine();
  }

  // Save all meshes in the mesh hierarchy in VTK format
  File file("mesh3D.pvd");
  MeshHierarchy meshes(mesh);
  for (MeshIterator m(meshes); !m.end(); ++m)
    file << *m;
}
