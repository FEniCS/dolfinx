// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __CELL_H
#define __CELL_H

#include <dolfin/dolfin_log.h>
#include <dolfin/CellIterator.h>
#include <dolfin/NodeIterator.h>
#include <dolfin/EdgeIterator.h>

namespace dolfin {  

  class Point;
  class Node;
  class Edge;
  class Triangle;
  class Tetrahedron;
  
  class Cell {
  public:

	 enum Type   { TRIANGLE, TETRAHEDRON, NONE };
	 enum Marker { MARKED_FOR_REGULAR_REFINEMENT, MARKED_FOR_IRREGULAR_REFINEMENT, 
		       MARKED_FOR_NO_REFINEMENT, MARKED_FOR_COARSENING, MARKED_ACCORDING_TO_REFINEMENT };
	 enum Status { REFINED_REGULARLY, REFINED_IRREGULARLY, UNREFINED };
	 
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
	 Edge* edge(int i) const;
	 Cell* neighbor(int i) const;
	 Cell* son(int i) const;
	 Point coord(int i) const;
	 Type  type() const;
	 int   noCellNeighbors() const;
	 int   noNodeNeighbors() const;

	 // id information for cell and its contents
	 int id() const;
	 int nodeID(int i) const;
	 int level() const;

	 // Mark and check state of the marke
	 void mark(Marker marker);
	 Marker marker() const;

	 void mark_edge(int edge);
	 void unmark_edge(int edge);
	 int noMarkedEdges();
	 bool markedEdgesOnSameFace();
	 
	 void setStatus(Status status);
	 Status status() const; 

	 // -> access passed to GenericCell
	 GenericCell* operator->() const;
	 
	 /// Output
	 friend LogStream& operator<<(LogStream& stream, const Cell& cell);
	 
	 // Friends
	 friend class GridData;
	 friend class InitGrid;
	 friend class NodeIterator::CellNodeIterator;
	 friend class CellIterator::CellCellIterator;
	 friend class EdgeIterator::CellEdgeIterator;
	 friend class Triangle;
	 friend class Tetrahedron;
	 
  private:

	 void set(Node *n0, Node *n1, Node *n2);
	 void set(Node *n0, Node *n1, Node *n2, Node *n3);
	 
	 void setLevel(int level);
	 void setID(int id);
	 void init(Type type);
	 bool neighbor(Cell &cell);

	 // Global cell number
	 int _id;
	 
	 // Refinement level in grid hierarchy, coarsest grid is level = 0
	 int _level;

	 // Refinement status
	 Status _status;

	 // Marker (for refinement)
	 Marker _marker;
	 int _no_marked_edges;

	 // The cell
	 GenericCell *c;
	 
	 // Connectivity
	 ShortList<Cell *> cc;
	 ShortList<Node *> cn;
	 ShortList<Edge *> ce;
	 ShortList<Cell *> children;

  };

}

#endif
