// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// A couple of comments:
//
//   - Enums don't need to be called "refined_regular", just "regular" is enough?
//   - Move data to GenericCell? Cell should only contain the GenericCell pointer?
//   - Maybe more methods should be private?
//
// Modified by Anders Logg, 2005.

#ifndef __CELL_H
#define __CELL_H

#include <dolfin/dolfin_log.h>
#include <dolfin/PArray.h>
#include <dolfin/CellIterator.h>
#include <dolfin/NodeIterator.h>
#include <dolfin/EdgeIterator.h>
#include <dolfin/FaceIterator.h>

namespace dolfin
{
  
  class Point;
  class Node;
  class Edge;
  class Triangle;
  class Tetrahedron;
  class Mesh;
  class MeshInit;
  class CellRefData;
  
  class Cell
  {
  public:

    /// Cell types (triangle or tetrahedron)
    enum Type { triangle, tetrahedron, none };

    /// Orientation (left or right), note that right means counter-clockwise for a triangle
    enum Orientation { left, right };

    /// Create an empty cell
    Cell();

    /// Create cell (triangle) from three given nodes
    Cell(Node& n0, Node& n1, Node& n2);

    /// Create cell (tetrahedron) from four given nodes
    Cell(Node& n0, Node& n1, Node& n2, Node& n3);

    /// Destructor
    ~Cell();

    /// Clear cell data
    void clear();

    ///--- Cell data ---

    /// Return id of cell
    int id() const;

    /// Return cell type
    Type type() const;

    /// Return orientation of cell
    Orientation orientation() const;
    
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
    Node& node(int i) const;

    /// Return edge number i
    Edge& edge(int i) const;

    /// Return face number i
    Face& face(int i) const;

    /// Return cell neighbor number i
    Cell& neighbor(int i) const;

    /// Return parent cell (null if no parent)
    Cell* parent() const;

    /// Return child cell (null if no child)
    Cell* child(int i) const;

    /// Return the mesh containing the cell
    Mesh& mesh();
    
    /// Return the mesh containing the cell (const version)
    const Mesh& mesh() const;

    /// Return coordinate for node i
    Point& coord(int i) const;

    /// Return midpoint of cell
    Point midpoint() const;

    // Return id for node number i
    int nodeID(int i) const;
    
    // Compute and return volume / area
    real volume() const;

    // Compute and return diameter 
    real diameter() const;

    /// Comparison with another cell
    bool operator==(const Cell& cell) const;

    /// Comparison with another cell
    bool operator!=(const Cell& cell) const;
    
    ///--- Mesh refinement ---

    /// Mark cell for refinement
    void mark();
    
    ///--- Output ---

    /// Display condensed cell data
    friend LogStream& operator<<(LogStream& stream, const Cell& cell);
    
    // Friends
    friend class Mesh;
    friend class MeshData;
    friend class MeshInit;
    friend class MeshRefinement;
    friend class MeshRefinementData;
    friend class TriMeshRefinement;
    friend class TetMeshRefinement;
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
    int setID(int id, Mesh& mesh);
    
    // Set the mesh pointer
    void setMesh(Mesh& mesh);

    // Set parent cell
    void setParent(Cell& parent);

    // Remove parent cell
    void removeParent();

    // Set number of children
    void initChildren(int n);

    // Set child cell
    void addChild(Cell& child);

    // Remove child cell
    void removeChild(Cell& child);

    // Clear data and create a new triangle
    void set(Node& n0, Node& n1, Node& n2);

    // Clear data and create a new tetrahedron
    void set(Node& n0, Node& n1, Node& n2, Node& n3);

    // Check if given cell is a neighbor
    bool neighbor(Cell& cell) const;

    // Check if given node is contained in the cell
    bool haveNode(Node& node) const;

    // Check if given edge is contained in the cell
    bool haveEdge(Edge& edge) const;

    // Create edges for the cell
    void createEdges();

    // Create faces for the cell
    void createFaces();

    // Create a given edge
    void createEdge(Node& n0, Node& n1);

    // Create a given face
    void createFace(Edge& e0, Edge& e1, Edge& e2);

    // Find node with given coordinates (null if not found)
    Node* findNode(const Point& p) const;
       
    // Find edge within cell (null if not found)
    Edge* findEdge(Node& n0, Node& n1);

    // Find face within cell (null if not found)
    Face* findFace(Edge& e0, Edge& e1, Edge& e2);
    Face* findFace(Edge& e0, Edge& e1);

    // Return cell marker
    Marker& marker();

    // Return cell status
    Status& status();

    // Sort mesh entities locally
    void sort();

    //--- Cell data ---

    // The cell
    GenericCell* c;
    
  };

}

#endif
