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
  Mesh mesh("mesh2D.xml.gz");
  //Mesh mesh("mesh3D.xml.gz");

  //mesh.refine();



  MeshFunction<bool> cell_marker(mesh);
  cell_marker.init(mesh.topology().dim());
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    cout << "cell " << c->index() << ": midpoint (" << c->midpoint().x() << 
      "," << c->midpoint().y() << "," << c->midpoint().z() << ")" << endl;
    if ( fabs(c->midpoint().x()-0.75) < 0.2 ) cell_marker.set(c->index(),true);
    else                                 cell_marker.set(c->index(),false);
  }
  
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    cout << "cell " << cell->index() << ": " << cell_marker.get(*cell) << endl;

  LocalMeshRefinement::refineSimplexMeshByBisection(mesh,cell_marker);

  File mesh_file_fine("mesh-fine.pvd"); 
  mesh_file_fine << mesh; 

  MeshFunction<bool> cell_marker_coarsen(mesh);
  cell_marker_coarsen.init(mesh.topology().dim());
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    if ( fabs(c->midpoint().x()-0.75) < 0.2 ) cell_marker_coarsen.set(c->index(),true);
    else                                      cell_marker_coarsen.set(c->index(),false);
  }
  
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    cout << "cell " << cell->index() << ": " << cell_marker_coarsen.get(*cell) << endl;

  LocalMeshCoarsening::coarsenSimplexMeshByEdgeCollapse(mesh,cell_marker_coarsen);

  File mesh_file_coarse("mesh-coarse.pvd"); 
  mesh_file_coarse << mesh; 
  

  /*

  MeshFunction<bool> cell_marker_2(mesh);
  cell_marker_2.init(mesh.topology().dim());
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    if ( fabs(c->midpoint().x()-0.75) < 0.2 ) cell_marker_2.set(c->index(),true);
    else                                 cell_marker_2.set(c->index(),false);
  }
  


  LocalMeshRefinement::refineSimplexMeshByBisection(mesh,cell_marker_2);

  MeshFunction<bool> cell_marker_3(mesh);
  cell_marker_3.init(mesh.topology().dim());
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    if ( fabs(c->midpoint().x()-0.75) < 0.2 ) cell_marker_3.set(c->index(),true);
    else                                 cell_marker_3.set(c->index(),false);
  }
  


  LocalMeshRefinement::refineSimplexMeshByBisection(mesh,cell_marker_3);

  MeshFunction<bool> cell_marker_4(mesh);
  cell_marker_4.init(mesh.topology().dim());
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    if ( fabs(c->midpoint().x()-0.75) < 0.2 ) cell_marker_4.set(c->index(),true);
    else                                 cell_marker_4.set(c->index(),false);
  }
  


  LocalMeshRefinement::refineSimplexMeshByBisection(mesh,cell_marker_4);
  */


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
