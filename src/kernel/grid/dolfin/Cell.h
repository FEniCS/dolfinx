// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// A couple of comments:
//
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
  class CellRefData;
  
  class Cell {
  public:

    enum Type   { triangle, tetrahedron, none };

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

    /// Return number of cell children
    int noChildren() const;

    /// Return node number i
    Node* node(int i) const;

    /// Return edge number i
    Edge* edge(int i) const;

    /// Return cell neighbor number i
    Cell* neighbor(int i) const;

    /// Return parent cell
    Cell* parent() const;

    /// Return child cell
    Cell* child(int i) const;

    /// Return coordinate for node i
    Point coord(int i) const;

    /// Return midpoint of cell
    Point midpoint() const;

    // Return id for node number i
    int nodeID(int i) const;

    ///--- Grid refinement ---

    /// Mark cell for refinement
    void mark();
    
    ///--- Output ---

    /// Display condensed cell data
    friend LogStream& operator<<(LogStream& stream, const Cell& cell);
    
    // Friends
    friend class Grid;
    friend class GridData;
    friend class GridInit;
    friend class GridRefinement;
    friend class GridRefinementData;
    friend class TriGridRefinement;
    friend class TetGridRefinement;
    friend class GenericCell;
    friend class CellRefData;
    friend class NodeIterator::CellNodeIterator;
    friend class CellIterator::CellCellIterator;
    friend class EdgeIterator::CellEdgeIterator;
    friend class FaceIterator::CellFaceIterator;
    
  private:
    
    enum Marker { marked_for_reg_ref,        // Marked for regular refinement
		  marked_for_irr_ref_1,      // Marked for irregular refinement by rule 1
		  marked_for_irr_ref_2,      // Marked for irregular refinement by rule 2
		  marked_for_irr_ref_3,      // Marked for irregular refinement by rule 3
		  marked_for_irr_ref_4,      // Marked for irregular refinement by rule 4
		  marked_for_no_ref,         // Marked for no refinement
		  marked_for_coarsening,     // Marked for coarsening
		  marked_according_to_ref }; // Marked according to refinement
    
    enum Status { ref_reg,                   // Refined regularly
		  ref_irr,                   // Refined irregularly
		  unref };                   // Unrefined
    
    // Specify global cell number
    int setID(int id, Grid* grid);
    
    // Set parent cell
    void setParent(Cell* parent);

    // Set child cell
    void addChild(Cell* child);

    // Clear data and create a new triangle
    void set(Node* n0, Node* n1, Node* n2);

    // Clear data and create a new tetrahedron
    void set(Node* n0, Node* n1, Node* n2, Node* n3);

    // Check if given cell is a neighbor
    bool neighbor(Cell& cell) const;

    // Check if given edge is contained in the cell
    bool haveEdge(Edge& edge) const;

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

    // Return cell marker
    Marker& marker();

    // Return cell status
    Status& status();

    //--- Cell data ---

    // The cell
    GenericCell* c;
    
  };

}

#endif
