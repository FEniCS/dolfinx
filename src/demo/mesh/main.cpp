// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

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
  
  // Save first mesh
  File file("meshes2D.m");
  file << mesh;

  // Refine a couple of times
  for (int i = 0; i < refinements; i++)
  {
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      // Mark cells close to y = x
      for (NodeIterator node(cell); !node.end(); ++node)
	if ( fabs(node->coord().x - node->coord().y) < DOLFIN_EPS )
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

    // Save all meshes in the mesh hierarcy
    MeshHierarchy meshes(mesh);
    for (MeshIterator g(meshes); !g.end(); ++g)
      file << *g;

  }

  dolfin::cout << dolfin::endl;
}

void refine3D(int refinements)
{
  dolfin::cout << "Refining 3D mesh" << dolfin::endl;
  dolfin::cout << "----------------" << dolfin::endl;  

  // Load mesh
  Mesh mesh("mesh3D.xml.gz");
  
  // Save original mesh in OpenDX format
  File unref("mesh3D_unrefined.dx");
  unref << mesh;

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

  // Save refined mesh in OpenDX format
  File ref("mesh3D_refined.dx");
  ref << mesh;
}
