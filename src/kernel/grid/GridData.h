// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GRID_DATA_H
#define __GRID_DATA_H

/// GridData is a container for grid data.
///
/// Block linked list is used to store the grid data,
/// constisting of
///
///    a list of all nodes
///    a list of all cells

#include <List.h>

namespace dolfin {

  class Node;
  class Triangle;
  class Tetrahedron;
  
  class GridData {
  public:
	 
	 GridData();
	 ~GridData();
	 
	 Node*        createNode();
	 Triangle*    createTriangle();
	 Tetrahedron* createTetrahedron();
	 
	 Node*        createNode(real x, real y, real z);
	 Triangle*    createTriangle(int n0, int n1, int n2);
	 Tetrahedron* createTetrahedron(int n0, int n1, int n2, int n3);
	 
	 Node*        getNode(int id);
	 Triangle*    getTriangle(int id);
	 Tetrahedron* getTetrahedron(int id);
	 
  private:
	 
	 List<Node> nodes;
	 List<Triangle> triangles;
	 List<Tetrahedron> tetrahedrons;
	 
  };

}
  
#endif
