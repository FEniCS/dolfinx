// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GENERIC_CELL_H
#define __GENERIC_CELL_H

#include <dolfin/constants.h>
#include <dolfin/Cell.h>
#include <dolfin/NodeIterator.h>
#include <dolfin/CellIterator.h>
#include <dolfin/EdgeIterator.h>
#include <dolfin/FaceIterator.h>
#include <dolfin/Array.h>

namespace dolfin {

  class Point;
  class Node;
  class Cell;
  class Grid;
  
  class GenericCell {
  public:
	 
    GenericCell();
    virtual ~GenericCell();

    int id() const;
    virtual Cell::Type type() const = 0;

    virtual int noNodes() const = 0;
    virtual int noEdges() const = 0;
    virtual int noFaces() const = 0;
    virtual int noBoundaries() const = 0;
    
    int noCellNeighbors() const;
    int noNodeNeighbors() const;

    Node* node(int i) const;
    Edge* edge(int i) const;
    Cell* neighbor(int i) const;
    Point coord(int i) const;
    Point midpoint() const;
    int   nodeID(int i) const;

    void mark(Cell* cell);
    
    // Friends
    friend class Cell;
    friend class Triangle;
    friend class Tetrahedron;
    friend class GridInit;
    friend class NodeIterator::CellNodeIterator;
    friend class CellIterator::CellCellIterator;
    friend class EdgeIterator::CellEdgeIterator;
    friend class FaceIterator::CellFaceIterator;
    
  private:

    // Specify global cell number
    int setID(int id, Grid* grid);
    
    // Check if given cell is a neighbor
    bool neighbor(GenericCell* cell) const;

    // Create edges for the cell
    virtual void createEdges() = 0;

    // Create faces for the cell
    virtual void createFaces() = 0;

    // Create a given edge
    void createEdge(Node* n0, Node* n1);

    // Create a given face
    void createFace(Edge* e0, Edge* e1, Edge* e2);

    // Find edge within cell
    Edge* findEdge(Node* n0, Node* n1);

    // Find face within cell
    Face* findFace(Edge* e0, Edge* e1, Edge* e2);

    // The grid containing this cell
    Grid* grid;

    // Global cell number
    int _id;

    // Connectivity
    Array<Node*> cn;
    Array<Cell*> cc;
    Array<Edge*> ce;
    Array<Face*> cf;

    // FIXME: Remove?
    Array<Cell*> children;    
    
  };

}

#endif
