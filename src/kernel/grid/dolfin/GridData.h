// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// A couple of comments:
//
//   - Is the check in createNode() really necessary?

#ifndef __GRID_DATA_H
#define __GRID_DATA_H

#include <dolfin/Table.h>
#include <dolfin/Node.h>
#include <dolfin/Cell.h>
#include <dolfin/Edge.h>
#include <dolfin/Face.h>

namespace dolfin {

  /// GridData is a container for grid data.
  ///
  /// A Table (block-linked list) is used to store the grid data:
  ///
  ///   a table of all nodes (n)
  ///   a table of all cells (c)
  ///   a table of all edges (e)
  ///   a table of all faces (f)
  ///
  /// Connectivity is stored locally. With four different geometric
  /// objects (n, c, e, f), 16 different combinations are
  /// possible. The combinations marked with an (x) are computed and
  /// stored:
  ///
  ///   (x) n-n (the node neighbors of a node)  [5, from 4]
  ///   (x) n-c (the cell neighbors of a node)  [1, from 0]
  ///   (x) n-e (the edge neighbors of a node)  [4, from 3]
  ///       n-f (the face neighbors of a node)
  ///
  ///   (x) c-n (the nodes within a cell)       [0]
  ///   (x) c-c (the cell neighbors of a cell)  [2, from 0 and 1]
  ///   (x) c-e (the edges within a cell)       [3, from 0 and 2]
  ///   (x) c-f (the faces within a cell)       [6, from 0 and 2]
  ///
  ///   (x) e-n (the nodes within an edge)      [3, from 0 and 2]
  ///       e-c (the cell neighbors of an edge)
  ///       e-e (the edge neighbors of an edge)
  ///       e-f (the face neighbors of an edge)
  ///
  ///       f-n (the nodes within a face)
  ///       f-c (the cell neighbors of a face)
  ///   (x) f-e (the edges within a face)       [6, from 1 and 3]
  ///       f-f (the face neighbors of a face)
  ///
  /// The numbers within brackets indicate in which order the
  /// connectivity is computed. A [0] indicates that the information
  /// is known a priori.

  class GridData {
  public:
    
    /// Create an empty set of grid data
    GridData(Grid* grid);

    /// Destructor
    ~GridData();

    /// Clear all data
    void clear();

    Node* createNode(Point p);
    Node* createNode(real x, real y, real z);
    
    Cell* createCell(int n0, int n1, int n2);
    Cell* createCell(int n0, int n1, int n2, int n3);
    Cell* createCell(Node* n0, Node* n1, Node* n2);
    Cell* createCell(Node* n0, Node* n1, Node* n2, Node* n3);

    Edge* createEdge(int n0, int n1);
    Edge* createEdge(Node* n0, Node* n1);

    Face* createFace(int e0, int e1, int e2);
    Face* createFace(Edge* e0, Edge* e1, Edge* e2);
    
    Node* getNode(int id);
    Cell* getCell(int id);
    Edge* getEdge(int id);
    Face* getFace(int id);
    
    int noNodes() const;
    int noCells() const;
    int noEdges() const;
    int noFaces() const;
    
    bool hasEdge(Node* n0, Node* n1) const;

    // Friends
    friend class NodeIterator::GridNodeIterator;
    friend class CellIterator::GridCellIterator;
    friend class EdgeIterator::GridEdgeIterator;
    friend class FaceIterator::GridFaceIterator;
    
  private:
    
    // The grid
    Grid* grid;

    // Table of all nodes within the grid
    Table<Node> nodes;

    // Table of all cells within the grid
    Table<Cell> cells;

    // Table of all edges within the grid
    Table<Edge> edges;

    // Table of all faces within the grid
    Table<Face> faces;
   
  };
  
}
  
#endif
