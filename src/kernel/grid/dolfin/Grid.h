// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GRID_H
#define __GRID_H

#include <dolfin/dolfin_log.h>
#include <dolfin/Variable.h>
#include <dolfin/constants.h>
#include <dolfin/List.h>
#include <dolfin/Point.h>
#include <dolfin/Node.h>
#include <dolfin/Cell.h>
#include <dolfin/Edge.h>
#include <dolfin/Face.h>
#include <dolfin/BoundaryData.h>
#include <dolfin/GridData.h>
#include <dolfin/GridRefinementData.h>

namespace dolfin {

  class GridData;
  
  class Grid : public Variable {
  public:
    
    enum Type { triangles, tetrahedrons };
    
    /// Create an empty grid
    Grid();

    /// Create grid from given file
    Grid(const char *filename);

    /// Destructor
    ~Grid();

    ///--- Basic functions

    /// Clear grid
    void clear();

    /// Return number of nodes in the grid
    int noNodes() const;

    /// Return number of cells in the grid
    int noCells() const;

    /// Return number of edges in the grid
    int noEdges() const;

    /// Return number of faces in the grid
    int noFaces() const;
    
    /// Return type of grid
    Type type() const;

    ///--- Grid refinement ---

    /// Mark cell for refinement
    void mark(Cell* cell);

    /// Refine grid
    void refine();
    
    ///--- Output ---

    /// Display grid data
    void show();

    /// Display condensed grid data
    friend LogStream& operator<< (LogStream& stream, const Grid& grid);
    
    /// Friends
    friend class GenericCell;
    friend class NodeIterator::GridNodeIterator;
    friend class NodeIterator::BoundaryNodeIterator;
    friend class CellIterator::GridCellIterator;
    friend class EdgeIterator::GridEdgeIterator;
    friend class FaceIterator::GridFaceIterator;
    friend class XMLGrid;
    friend class GridInit;
    friend class GridRefinement;
    friend class GridHierarchy;
    friend class Boundary;
    friend class BoundaryInit;
    
  private:
    
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

    bool hasEdge(Node* n0, Node* n1) const;
    
    void init();
    
    // Grid data
    GridData gd;

    // Boundary data
    BoundaryData bd;

    // Grid refinement data
    GridRefinementData rd;
    
    // Parent grid
    Grid* _parent;

    // Child grid
    Grid* _child;
    
    // Grid type
    Type _type;

  };
  
}

#endif
