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
#include <dolfin/Array.h>
#include <dolfin/CellIterator.h>
#include <dolfin/NodeIterator.h>
#include <dolfin/EdgeIterator.h>
#include <dolfin/FaceIterator.h>

namespace dolfin {  

  class Point;
  class Node;
  class Edge;
  class Triangle;
  class Tetrahedron;
  class Grid;
  class GridInit;
  
  class Cell {
  public:
    
    enum Type   { triangle, tetrahedron, none };

    enum Marker { MARKED_FOR_REGULAR_REFINEMENT, MARKED_FOR_IRREGULAR_REFINEMENT, 
		     MARKED_FOR_IRREGULAR_REFINEMENT_BY_1, MARKED_FOR_IRREGULAR_REFINEMENT_BY_2, 
		     MARKED_FOR_IRREGULAR_REFINEMENT_BY_3, MARKED_FOR_IRREGULAR_REFINEMENT_BY_4, 
		     MARKED_FOR_NO_REFINEMENT, MARKED_FOR_COARSENING, MARKED_ACCORDING_TO_REFINEMENT };
    enum Status { REFINED_REGULAR, REFINED_IRREGULAR, 
		  REFINED_IRREGULAR_BY_1, REFINED_IRREGULAR_BY_2, 
		  REFINED_IRREGULAR_BY_3, REFINED_IRREGULAR_BY_4, UNREFINED };
    
    /// Create an empty cell
    Cell();

    /// Create cell (triangle) from three given nodes
    Cell(Node* n0, Node* n1, Node* n2);

    /// Create cell (tetrahedron) from four given nodes
    Cell(Node* n0, Node* n1, Node* n2, Node* n3);

    /// Destructor
    ~Cell();
    
    ///--- Cell data ---

    /// Return id of cell
    int id() const;

    /// Return cell type
    Type type() const;

    /// Return number of nodes
    int noNodes() const;

    /// Return number of edges
    int noEdges() const;

    /// Return number of faces
    int noFaces() const;

    /// Return number of boundaries
    int noBoundaries() const;

    /// Return number of cell neighbors
    int noCellNeighbors() const;

    /// Return number of node neighbors
    int noNodeNeighbors() const;

    /// Return node number i
    Node* node(int i) const;

    /// Return edge number i
    Edge* edge(int i) const;

    /// Return cell neighbor number i
    Cell* neighbor(int i) const;

    /// Return coordinate for node i
    Point coord(int i) const;

    /// Return midpoint of cell
    Point midpoint() const;

    // Return id for node number i
    int nodeID(int i) const;

    ///--- Grid refinement ---

    /// Mark cell for refinement
    void mark();


    
    // FIXME: Remove?
    
    /*
      void addChild(Cell* child);
      void setEdge(Edge* e, int i);
      int level() const;
      void setLevel(int level);
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
    */
    
    ///--- Output ---

    /// Display condensed cell data
    friend LogStream& operator<<(LogStream& stream, const Cell& cell);
    
    // Friends
    friend class GridData;
    friend class GridInit;
    friend class GridRefinement;
    friend class GenericCell;
    friend class NodeIterator::CellNodeIterator;
    friend class CellIterator::CellCellIterator;
    friend class EdgeIterator::CellEdgeIterator;
    friend class FaceIterator::CellFaceIterator;
    
  private:

    // Specify global cell number
    int setID(int id, Grid* grid);
    
    // Clear data and create a new triangle
    void set(Node* n0, Node* n1, Node* n2);

    // Clear data and create a new tetrahedron
    void set(Node* n0, Node* n1, Node* n2, Node* n3);

    // Check if given cell is a neighbor
    bool neighbor(Cell& cell);

    // Create edges for the cell
    void createEdges();

    // Create faces for the cell
    void createFaces();

    // Create a given edge
    void createEdge(Node* n0, Node* n1);

    // Create a given face
    void createFace(Edge* e0, Edge* e1, Edge* e2);
       
    // Find edge within cell
    Edge* findEdge(Node* n0, Node* n1);

    // Find face within cell
    Face* findFace(Edge* e0, Edge* e1, Edge* e2);

    // The cell
    GenericCell* c;

    // FIXME: Remove?

    // Refinement level in grid hierarchy, coarsest grid is level = 0
    int _level;
    // Refinement status
    Status _status;
    // Marker (for refinement)
    Marker _marker;
    int _no_marked_edges;
    bool _marked_for_re_use;
    bool _refined_by_face_rule;

    
  };

}

#endif
