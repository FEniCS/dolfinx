// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GRID_HH
#define __GRID_HH

// FIXME: remove
#include <stdio.h>

#include <iostream>

#include <dolfin/dolfin_constants.h>
#include <dolfin/List.h>

namespace dolfin{

  class Node;
  class Cell;
  class Triangle;
  class Tetrahedron;
  class GridData;
  
  class Grid{
  public:
	 
	 Grid();
	 Grid(const char *filename);
	 ~Grid();

	 void clear();

	 int noNodes();
	 int noCells();

	 /// Output
	 void show();
	 friend ostream& operator << (ostream& output, Grid& grid);
	 
	 /// Friends
	 friend class NodeIterator;
	 friend class XMLGrid;
	 
	 // old functions below
	 
	 /// Compute info about neighbors
	 void Init();
	 /// Clear info about neighbors
	 void Clear();
	 
	 /// Get number of nodes
	 int GetNoNodes();
	 /// Get number of cells
	 int GetNoCells();
	 /// Get maximum number of nodes in all cells
	 int GetMaxCellSize();
	 /// True if all cells are of the same type
	 bool AllCellsEqual();
	 /// Get diameter of smallest cell
	 real GetSmallestDiameter();
  
	 /// Get node from node number
	 Node* GetNode(int node);
	 /// Get cell from cell number
	 Cell* GetCell(int cell);
	 
	 /// Display information about the grid
	 void Display();
	 /// Display the whole grid
	 void DisplayAll();
	 /// Read grid from file (type determined from prefix)
	 void Read(const char *filename);
	 /// Save grid to file (type determined from prefix)
	 void Write(const char *filename);
	 
  private:
	 
 	 Node*        createNode();
	 Triangle*    createTriangle();
	 Tetrahedron* createTetrahedron();
	 
	 Node*        createNode(real x, real y, real z);
	 Triangle*    createTriangle(int n0, int n1, int n2);
	 Tetrahedron* createTetrahedron(int n0, int n1, int n2, int n3);

	 Node*        getNode(int id);
	 Triangle*    getTriangle(int id);
	 Tetrahedron* getTetrahedron(int id);

	 // old functions below
	 
	 
	 /// Save grid to inp file
	 void WriteINP(FILE *fp);
	 
	 /// Allocate memory for nodes
	 void AllocNodes(int newsize);
	 /// Allocate memory for cells
	 //  void AllocCells(int newsize, CellType newtype);
	 
	 /// Clear all information about neighbors
	 void ClearNeighborInfo();
	 
	 /// Compute maximum cell size
	 //  void ComputeMaximumCellSize();
	 /// Compute smallest cell diameter
	 // void ComputeSmallestDiameter();
	 /// Compute info about neighbors
	 //void ComputeNeighborInfo();
	 
	 /// --- Grid data ---
	 
	 /// Maximum cell size (maximum number of nodes in cells)
	 int maxcellsize;
	 /// Size of grid (bytes)
	 int mem;
	 /// Diameter of smallest cell
	 real h;
	 
	 /// Type of cells
	 //CellType celltype;
	 
	 /// --- Grid data (main part) ---

	 GridData *grid_data;
	 int no_nodes;
	 int no_cells;

	 // old data
	 
	 /// Nodes (positions)
	 Node *nodes;
	 /// Cells (connections)
	 Cell **cells;
	 
  };

  // Iterators

  class NodeIterator {
  public:

	 NodeIterator(Grid& grid); 

	 NodeIterator& operator++();
	 bool end();
	 
	 Node  operator*() const;
	 Node* operator->() const;
	 bool  operator==(const NodeIterator& n) const;
	 bool  operator!=(const NodeIterator& n) const;

	 
  private:

	 List<Node>::Iterator node_iterator;
	 List<Node>::Iterator at_end;
	 
  };
  
}

#endif
