// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __CELL_H
#define __CELL_H

#include <dolfin/dolfin_log.h>
#include <dolfin/Point.h>
#include <dolfin/CellIterator.h>
#include <dolfin/NodeIterator.h>

namespace dolfin {  

  class Node;
  class Triangle;
  class Tetrahedron;
  
  class Cell {
  public:

	 enum Type   { TRIANGLE, TETRAHEDRON, NONE };
	 enum Marker { MARKED, UNMARKED };
	 
	 Cell();
	 Cell(Node &n0, Node &n1, Node &n2);
	 Cell(Node &n0, Node &n1, Node &n2, Node &n3);
	 ~Cell();

	 // Number of nodes, edges, faces, boundaries
	 int noNodes() const;
	 int noEdges() const;
	 int noFaces() const;
	 int noBound() const; 

	 // Cell data
	 Node* node(int i) const;
	 Point coord(int i) const;
	 Type  type() const;
	 int   noCellNeighbors() const;
	 int   noNodeNeighbors() const;

	 // id information for cell and its contents
	 int id() const;
	 int nodeID(int i) const;
	 int edgeID(int i) const;

	 // Mark and check if marked
	 void mark();
	 bool marked() const;

	 // Mark and check state of the marke
	 void mark(Marker marker);
	 Marker marker() const;
	 
	 // -> access passed to GenericCell
	 GenericCell* operator->() const;
	 
	 /// Output
	 friend LogStream& operator<<(LogStream& stream, const Cell& cell);
	 
	 // Friends
	 friend class GridData;
	 friend class InitGrid;
	 friend class NodeIterator::CellNodeIterator;
	 friend class CellIterator::CellCellIterator;
	 friend class Triangle;
	 friend class Tetrahedron;
	 
  private:

	 void set(Node *n0, Node *n1, Node *n2);
	 void set(Node *n0, Node *n1, Node *n2, Node *n3);
	 
	 void setID(int id);
	 void init(Type type);
	 bool neighbor(Cell &cell);

	 // Global cell number
	 int _id;
	 
	 // The cell
	 GenericCell *c;

	 // Marker (for refinement)
	 Marker _marker;
	 
	 // Connectivity
	 ShortList<Cell *> cc;
	 ShortList<Node *> cn;

  };

}

#endif
