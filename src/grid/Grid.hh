// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GRID_HH
#define __GRID_HH

#include <stdio.h>

#include "Node.hh"
#include "Cell.hh"
#include "CellType.hh"
#include "GridData.hh"

class Grid{
public:

  Grid();
  ~Grid();
 
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
  
  /// Save grid to inp file
  void WriteINP(FILE *fp);

  /// Allocate memory for nodes
  void AllocNodes(int newsize);
  /// Allocate memory for cells
  void AllocCells(int newsize, CellType newtype);

  /// Clear all information about neighbors
  void ClearNeighborInfo();
  
  /// Compute maximum cell size
  void ComputeMaximumCellSize();
  /// Compute smallest cell diameter
  void ComputeSmallestDiameter();
  /// Compute info about neighbors
  void ComputeNeighborInfo();
  
  /// --- Grid data ---

  /// Number of nodes
  int no_nodes;
  /// Number of cells
  int no_cells;
  /// Maximum cell size (maximum number of nodes in cells)
  int maxcellsize;
  /// Size of grid (bytes)
  int mem;
  /// Diameter of smallest cell
  real h;
  
  /// Type of cells
  CellType celltype;

  /// --- Grid data (main part) ---

  GridData *gd;
  
  /// Nodes (positions)
  Node *nodes;
  /// Cells (connections)
  Cell **cells;

};

#endif
