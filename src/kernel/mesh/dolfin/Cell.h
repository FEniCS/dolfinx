// Copyright (C) 2002-2006 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2002
// Last changed: 2006-02-20
//
// A couple of comments:
//
//   - Enums don't need to be called "refined_regular", just "regular" is enough?
//   - Move data to GenericCell? Cell should only contain the GenericCell pointer?
//   - Maybe more methods should be private?

#ifndef __CELL_H
#define __CELL_H

#include <dolfin/dolfin_log.h>
#include <dolfin/PArray.h>
#include <dolfin/CellIterator.h>
#include <dolfin/VertexIterator.h>
#include <dolfin/EdgeIterator.h>
#include <dolfin/FaceIterator.h>

namespace dolfin
{
  
  class Point;
  class Vertex;
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

    /// Create cell (triangle) from three given vertices
    Cell(Vertex& n0, Vertex& n1, Vertex& n2);

    /// Create cell (tetrahedron) from four given vertices
    Cell(Vertex& n0, Vertex& n1, Vertex& n2, Vertex& n3);

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
    
    /// Return number of vertices
    int numVertices() const;

    /// Return number of edges
    int numEdges() const;

    /// Return number of faces
    int numFaces() const;

    /// Return number of boundaries
    int numBoundaries() const;

    /// Return number of cell neighbors
    int numCellNeighbors() const;

    /// Return number of vertex neighbors
    int numVertexNeighbors() const;

    /// Return number of cell children
    int numChildren() const;

    /// Return vertex number i
    Vertex& vertex(int i) const;

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

    /// Return coordinate for vertex i
    Point& coord(int i) const;

    /// Return midpoint of cell
    Point midpoint() const;

    // Return ID for vertex number i
    int vertexID(int i) const;

    // Return ID for edge number i
    int edgeID(int i) const;

    // Return ID for face number i
    int faceID(int i) const;
    
    // Compute and return volume / area
    real volume() const;

    // Compute and return diameter 
    real diameter() const;

    /// Compute alignment of given edge (0, 1)
    uint edgeAlignment(uint i) const;

    /// Compute alignment of given face (0, 1, 2, 3, 4, 5)
    uint faceAlignment(uint i) const;
    
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
    friend class VertexIterator::CellVertexIterator;
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
    void set(Vertex& n0, Vertex& n1, Vertex& n2);

    // Clear data and create a new tetrahedron
    void set(Vertex& n0, Vertex& n1, Vertex& n2, Vertex& n3);

    // Check if given cell is a neighbor
    bool neighbor(Cell& cell) const;

    // Check if given vertex is contained in the cell
    bool haveVertex(Vertex& vertex) const;

    // Check if given edge is contained in the cell
    bool haveEdge(Edge& edge) const;

    // Create edges for the cell
    void createEdges();

    // Create faces for the cell
    void createFaces();

    // Create a given edge
    void createEdge(Vertex& n0, Vertex& n1);

    // Create a given face
    void createFace(Edge& e0, Edge& e1, Edge& e2);

    // Find vertex with given coordinates (null if not found)
    Vertex* findVertex(const Point& p) const;
       
    // Find edge within cell (null if not found)
    Edge* findEdge(Vertex& n0, Vertex& n1);

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
