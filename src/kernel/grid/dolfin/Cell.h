// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// A couple of comments:
//
//   - Maybe we should not use capital letters for enums after all?
//   - Enums don't need to be called "refined_regular", just "regular" is enough?
//   - Move data to GenericCell? Cell should only contain the GenericCell pointer?
//   - Maybe more methods should be private?

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
		  MARKED_FOR_IRREGULAR_REFINEMENT_BY_1, MARKED_FOR_IRREGULAR_REFINEMENT_BY_2, 
		  MARKED_FOR_IRREGULAR_REFINEMENT_BY_3, MARKED_FOR_IRREGULAR_REFINEMENT_BY_4, 
		  MARKED_FOR_NO_REFINEMENT, MARKED_FOR_COARSENING, MARKED_ACCORDING_TO_REFINEMENT };
    enum Status { REFINED_REGULAR, REFINED_IRREGULAR, 
		  REFINED_IRREGULAR_BY_1, REFINED_IRREGULAR_BY_2, 
		  REFINED_IRREGULAR_BY_3, REFINED_IRREGULAR_BY_4, UNREFINED };
    
    Cell();
    Cell(Node &n0, Node &n1, Node &n2);
    Cell(Node &n0, Node &n1, Node &n2, Node &n3);
    ~Cell();
    
    // Number of nodes, edges, faces, boundaries
    int noNodes() const;
    int noEdges() const;
    int noFaces() const;
    int noBound() const; 
    int noChildren() const; 
    
    // Cell data
    Node* node(int i) const;
    Edge* edge(int i) const;
    Cell* neighbor(int i) const;
    Cell* child(int i) const;
    Point coord(int i) const;
    Type  type() const;
    int   noCellNeighbors() const;
    int   noNodeNeighbors() const;
    
    void addChild(Cell* child);
    
    void setEdge(Edge* e, int i);
    
    // id information for cell and its contents
    int id() const;
    int nodeID(int i) const;
    int level() const;
    void setLevel(int level);
    
    // Mark and check state of the marke
    void mark(Marker marker);
    Marker marker() const;
    
    void setMarkedForReUse(bool re_use);
    bool markedForReUse();
    
    void refineByFaceRule(bool refined_by_face_rule);
    bool refinedByFaceRule();
    
    void markEdge(int edge);
    void unmarkEdge(int edge);
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
    
    void setID(int id);
    void init(Type type);
    bool neighbor(Cell &cell);
    
   // Global cell number
    int _id;
    
    int _no_children;
    
    // Refinement level in grid hierarchy, coarsest grid is level = 0
    int _level;
    
    // Refinement status
    Status _status;
    
    // Marker (for refinement)
    Marker _marker;
    int _no_marked_edges;
    bool _marked_for_re_use;
    bool _refined_by_face_rule;
    
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
