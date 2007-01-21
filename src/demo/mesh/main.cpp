// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Hoffman 2006.
//
// First added:  2006-10-26
// Last changed: 2007-01-15

#include <dolfin.h>

using namespace dolfin;

int main()
{
  //UnitCube mesh(1,1,1);
  //UnitSquare mesh(1,1);
  //Mesh mesh("mesh2D.xml.gz");
  //Mesh mesh("mesh3D.xml.gz");
 Mesh mesh("dolfin.xml.gz");

  // Uniform refinement
  //mesh.refine();

  //mesh.disp();

  File mesh_file_0("mesh-0.pvd"); 
  mesh_file_0 << mesh; 

  unsigned int num_refinements = 3;
  for (unsigned int i = 0; i < num_refinements; i++)  
  {
    MeshFunction<bool> cell_marker(mesh);
    cell_marker.init(mesh.topology().dim());
    for (CellIterator c(mesh); !c.end(); ++c)
    {
      if ( fabs(c->midpoint().x()-0.75) < 0.9 ) cell_marker.set(c->index(),true);
      else                                      cell_marker.set(c->index(),false);
    }
    LocalMeshRefinement::refineMeshByEdgeBisection(mesh,cell_marker);
  }

  File mesh_file_fine("mesh-fine.pvd"); 
  mesh_file_fine << mesh; 

  unsigned int num_unrefinements = 3;
  for (unsigned int i = 0; i < num_unrefinements; i++)  
  {
    MeshFunction<bool> cell_marker(mesh);
    cell_marker.init(mesh.topology().dim());
    for (CellIterator c(mesh); !c.end(); ++c)
    {
      if ( fabs(c->midpoint().x()-0.75) < 0.9 ) cell_marker.set(c->index(),true);
      else                                      cell_marker.set(c->index(),false);
    }
    LocalMeshCoarsening::coarsenMeshByEdgeCollapse(mesh,cell_marker);
  }

  File mesh_file_coarse("mesh-coarse.pvd"); 
  mesh_file_coarse << mesh; 
  
  cout << "Iterating over the cells in the mesh..." << endl;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    cout << *cell << endl;
  
  BoundaryMesh boundary(mesh);  

  cout << "Iterating over the cells in the boundary..." << endl;
  for (CellIterator facet(boundary); !facet.end(); ++facet)
    cout << *facet << endl;

  File mesh_file("mesh-tst.pvd"); 
  mesh_file << mesh; 

  return 0;
}
