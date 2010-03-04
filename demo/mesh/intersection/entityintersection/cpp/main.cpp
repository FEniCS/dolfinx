// =====================================================================================
//
// Copyright (C) 2010-02-10  André Massing
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by André Massing, 2010
//
// First added:  2010-02-10
// Last changed: 2010-03-02
// 
//Author:  André Massing (am), massing@simula.no
//Company:  Simula Research Laboratory, Fornebu, Norway
//
// =====================================================================================

#include <dolfin.h> 

using namespace dolfin;

int main ()
{

#if HAS_CGAL
  UnitCube cube(3,3,2);
  cout <<"Total number of cells in Cube:" << cube.num_cells() <<endl;

  UnitSphere sphere(3);
  cout <<"Total number of cells in Sphere:" << sphere.num_cells() <<endl;
  cout <<"Intersecting pairwise cells of a cube and sphere mesh" << endl; 
  cout <<"Cube cell index | Sphere cell index" << endl;
  cout <<"------------------------------" << endl;

  for (CellIterator cube_cell(cube); !cube_cell.end(); ++cube_cell)
  {
    for (CellIterator sphere_cell(cube); !sphere_cell.end(); ++sphere_cell)
    {
      if (PrimitiveIntersector::do_intersect(*cube_cell, *sphere_cell))
	cout << cube_cell->index() << " | " << sphere_cell->index() << endl;
    }
  }
#else
  info("DOLFIN has been compiled without CGAL support.\nIntersetion functionality is not available");
 
#endif

  return 0;
}
