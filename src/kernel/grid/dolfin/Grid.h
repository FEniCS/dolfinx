// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GRID_H
#define __GRID_H

// FIXME: remove
#include <stdio.h>

#include <dolfin/dolfin_log.h>
#include <dolfin/Variable.h>
#include <dolfin/constants.h>
#include <dolfin/List.h>
#include <dolfin/Point.h>
#include <dolfin/Node.h>
#include <dolfin/Cell.h>
#include <dolfin/InitGrid.h>
#include <dolfin/RefineGrid.h>

namespace dolfin {

  class GridData;
  
  class Grid : public Variable {
  public:
    
    enum Type { TRIANGLES, TETRAHEDRONS };
    
    Grid();
    Grid(const char *filename);
    ~Grid();
    
    void operator = ( const Grid& grid );
    
    void clear();
    void refine();
    
    int  noNodes() const;
    int  noCells() const;
    Type type() const;
    
    /// Output
    void show();
    friend LogStream& operator<< (LogStream& stream, const Grid& grid);
    
    /// Friends
    friend class NodeIterator::GridNodeIterator;
    friend class CellIterator::GridCellIterator;
    friend class XMLGrid;
    friend class RefineGrid;
    
  private:
    
    Node* createNode();
    Cell* createCell(int level, Cell::Type type);
    Edge* createEdge();
    
    Node* createNode(Point p);
    Node* createNode(real x, real y, real z);

    Cell* createCell(int level, Cell::Type type, int n0, int n1, int n2);
    Cell* createCell(int level, Cell::Type type, int n0, int n1, int n2, int n3);
    Cell* createCell(int level, Cell::Type type, Node* n0, Node* n1, Node* n2);
    Cell* createCell(int level, Cell::Type type, Node* n0, Node* n1, Node* n2, Node* n3);
    
    Cell* createCell(Cell* parent, Cell::Type type, int n0, int n1, int n2);
    Cell* createCell(Cell* parent, Cell::Type type, int n0, int n1, int n2, int n3);
    Cell* createCell(Cell* parent, Cell::Type type, Node* n0, Node* n1, Node* n2);
    Cell* createCell(Cell* parent, Cell::Type type, Node* n0, Node* n1, Node* n2, Node* n3);

    Edge* createEdge(int n0, int n1);
    Edge* createEdge(Node* n0, Node* n1);

    Node* getNode(int id);
    Cell* getCell(int id);
    Edge* getEdge(int id);
    
    void init();
    
    /// Grid data
    GridData *gd;
    
    /// Grid type
    Type _type;
    
    /// Algorithms
    InitGrid initGrid;
    RefineGrid refineGrid;
    
  };
  
}

#endif
