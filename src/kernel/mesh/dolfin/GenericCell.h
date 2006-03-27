// Copyright (C) 2002-2006 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2002
// Last changed: 2006-03-24

#ifndef __GENERIC_CELL_H
#define __GENERIC_CELL_H

#include <dolfin/constants.h>
#include <dolfin/Cell.h>
#include <dolfin/VertexIterator.h>
#include <dolfin/CellIterator.h>
#include <dolfin/EdgeIterator.h>
#include <dolfin/FaceIterator.h>
#include <dolfin/PArray.h>

namespace dolfin
{

  class Point;
  class Vertex;
  class Cell;
  class Mesh;
  class CellRefData;
  
  class GenericCell
  {
  public:
	 
    GenericCell();
    virtual ~GenericCell();

    int id() const;
    virtual Cell::Type type() const = 0;
    virtual Cell::Orientation orientation() const = 0;

    virtual int numVertices() const = 0;
    virtual int numEdges() const = 0;
    virtual int numFaces() const = 0;
    virtual int numBoundaries() const = 0;
    
    int numCellNeighbors() const;
    int numVertexNeighbors() const;
    int numChildren() const;

    Vertex& vertex(int i) const;
    Edge& edge(int i) const;
    Face& face(int i) const;
    Cell& neighbor(int i) const;
    Cell* parent() const;
    Cell* child(int i) const;
    Point& coord(int i) const;
    Point midpoint() const;
    int vertexID(int i) const;
    int edgeID(int i) const;
    int faceID(int i) const;
    virtual real volume() const = 0;
    virtual real diameter() const = 0;
    virtual uint edgeAlignment(uint i) const = 0;
    virtual uint faceAlignment(uint i) const = 0;

    void mark(bool refine);

    // Friends
    friend class Cell;
    friend class MeshRefinement;
    friend class Triangle;
    friend class Tetrahedron;
    friend class MeshInit;
    friend class VertexIterator::CellVertexIterator;
    friend class CellIterator::CellCellIterator;
    friend class EdgeIterator::CellEdgeIterator;
    friend class FaceIterator::CellFaceIterator;
    
  private:

    // Specify global cell number
    int setID(int id, Mesh& mesh);
    
    // Set mesh pointer
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

    // Check if given cell is a neighbor
    bool neighbor(GenericCell& cell) const;

    // Check if given vertex is contained in the cell
    bool haveVertex(Vertex& vertex) const;

    // Check if given edge is contained in the cell
    bool haveEdge(Edge& edge) const;

    // Create edges for the cell
    virtual void createEdges() = 0;

    // Create faces for the cell
    virtual void createFaces() = 0;

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

    // Initialize marker (if not already done)
    void initMarker();

    // Return cell marker
    Cell::Marker& marker();

    // Return cell status
    Cell::Status& status();

    // Sort mesh entities locally
    virtual void sort() = 0;

    //--- Cell data ---

    // The mesh containing this cell
    Mesh* _mesh;

    // Global cell number
    int _id;

    // Connectivity
    PArray<Vertex*> cn;
    PArray<Cell*> cc;
    PArray<Edge*> ce;
    PArray<Face*> cf;

    // Parent-child info
    Cell* _parent;
    PArray<Cell*> children;

    // Mesh refinement data
    CellRefData* rd;
    
  };

}

#endif
