// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GRID_H
#define __GRID_H

// FIXME: remove
#include <stdio.h>

#include <iostream>

#include <dolfin/dolfin_constants.h>
#include <dolfin/List.h>
#include <dolfin/Node.h>
#include <dolfin/Cell.h>

namespace dolfin {

  class GridData;
  
  class Grid {
  public:
	 
	 Grid();
	 ~Grid();

	 void clear();

	 int noNodes();
	 int noCells();

	 /// Output
	 void show();
	 friend std::ostream& operator << (std::ostream& output, Grid& grid);
	 
	 /// Friends
	 friend class NodeIterator::GridNodeIterator;
	 friend class CellIterator::GridCellIterator;
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
	 
 	 Node* createNode();
	 Cell* createCell(Cell::Type type);

	 Node* createNode(real x, real y, real z);
	 Cell* createCell(Cell::Type type, int n0, int n1, int n2);
	 Cell* createCell(Cell::Type type, int n0, int n1, int n2, int n3);

	 Node* getNode(int id);
	 Cell* getCell(int id);

	 void init();
	 
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

}

#endif
