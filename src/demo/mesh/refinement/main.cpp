// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Hoffman 2006.
//
// First added:  2006-10-26
// Last changed: 2007-05-03

#include <dolfin.h>

using namespace dolfin;

int main()
{
  //UnitCube mesh(1,1,1);
  UnitSquare mesh(10,10);
  //Mesh mesh("mesh2D.xml.gz");
  //Mesh mesh("mesh3D.xml.gz");
  // Mesh mesh("dolfin.xml.gz");

  // Uniform refinement
  //mesh.refine();

  //mesh.disp();

  //Local mesh refinement
  File mesh_file_fine("mesh-fine.pvd"); 
  mesh_file_fine << mesh; 

  // Plot mesh
  plot(mesh);

  File mesh_file_coarse("mesh-coarse.pvd"); 
  mesh_file_coarse << mesh; 
  File mesh_file_coarse_xml("mesh-coarse.xml"); 
  //mesh_file_coarse_xml << mesh; 

  real t = 0.0;

  while(t < 2.0)
  {
    MeshFunction<bool> cell_refinement_marker(mesh);
    cell_refinement_marker.init(mesh.topology().dim());

    for (CellIterator c(mesh); !c.end(); ++c)
    {
      cell_refinement_marker.set(c->index(), false);

      if(fabs(c->midpoint().x() - t) < 0.1)
      {
	if(c->diameter() > 0.1)
	{
	  cout << "refining: " << endl;
	  cout << c->diameter() << endl;
	  cout << c->midpoint() << endl;
	  cout << c->index() << endl;

	  cell_refinement_marker.set(c->index(), true);
	}
      }
    }

    mesh.refine(cell_refinement_marker);
    //mesh.smooth();

    MeshFunction<bool> cell_derefinement_marker(mesh);
    cell_derefinement_marker.init(mesh.topology().dim());
    
    for (CellIterator c(mesh); !c.end(); ++c)
    {
      cell_derefinement_marker.set(c->index(), false);

      if(fabs(c->midpoint().x() - t) >= 0.1)
      {
	if(c->diameter() <= 0.1)
	{
	  cout << "coarsening: " << endl;
	  cout << c->diameter() << endl;
	  cout << c->midpoint() << endl;
	  cout << c->index() << endl;

	  cell_derefinement_marker.set(c->index(), true);
	}
      }
    }

    mesh.coarsen(cell_derefinement_marker);
    //mesh.smooth();

    mesh_file_coarse << mesh; 

    t += 0.1;
  }


  /*
  cout << "Iterating over the cells in the mesh..." << endl;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
    cout << *cell << endl;
  
  BoundaryMesh boundary(mesh);  

  cout << "Iterating over the cells in the boundary..." << endl;
  for (CellIterator facet(boundary); !facet.end(); ++facet)
    cout << *facet << endl;
  */

  //File mesh_file("mesh-tst.pvd"); 
  //mesh_file << mesh; 

  return 0;
}
