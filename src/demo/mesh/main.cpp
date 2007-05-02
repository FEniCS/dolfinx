// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Hoffman 2006.
//
// First added:  2006-10-26
// Last changed: 2007-01-15

#include <dolfin.h>

using namespace dolfin;

int main()
{
  UnitSquare mesh(1,1);

  // Uniform refinement
  //mesh.refine();

  //mesh.disp();

  // Local mesh refinement
  File file0("mesh.xml");
  file0 << mesh; 
  
  unsigned int num_refinements = 8;
  for (unsigned int i = 0; i < num_refinements; i++)  
  {
    MeshFunction<bool> cell_marker(mesh);
    cell_marker.init(mesh.topology().dim());
    for (CellIterator c(mesh); !c.end(); ++c)
    {
      if ( fabs(c->midpoint().x()-0.75) < 0.9 ) cell_marker.set(c->index(),true);
      else                                      cell_marker.set(c->index(),false);
    }
    mesh.refine(cell_marker);
  }

  // Local mesh refinement
  File file1("mesh_refined.xml");
  file1 << mesh; 

  unsigned int num_unrefinements = 2;
  for (unsigned int i = 0; i < num_unrefinements; i++)  
  {
    MeshFunction<bool> cell_marker(mesh);
    cell_marker.init(mesh.topology().dim());
    for (CellIterator c(mesh); !c.end(); ++c)
    {
      if ( fabs(c->midpoint().x()-0.75) < 0.1 ) cell_marker.set(c->index(),true);
      else                                      cell_marker.set(c->index(),false);
    }
    mesh.coarsen(cell_marker);
    //LocalMeshCoarsening::coarsenMeshByEdgeCollapse(mesh,cell_marker);
    
    mesh.smooth();
  }

  File file2("mesh_coarsened.xml");
  file2 << mesh; 

  // Extract boundary mesh
  BoundaryMesh boundary(mesh);
  File file3("mesh_boundary.xml");
  file3 << boundary;

  return 0;
}
