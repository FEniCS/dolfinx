// Copyright (C) 2002-2006 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2002
// Last changed: 2006-03-10

#ifndef __VERTEX_HH
#define __VERTEX_HH

#include <dolfin/dolfin_log.h>
#include <dolfin/constants.h>
#include <dolfin/PArray.h>
#include <set>
#include <dolfin/Point.h>
#include <dolfin/VertexIterator.h>
#include <dolfin/CellIterator.h>
#include <dolfin/EdgeIterator.h>

namespace dolfin
{

  class GenericCell;
  class Cell;
  class Edge;
  class MeshInit;
  
  class Vertex
  {
  public:

    /// Create an unconnected vertex at (0,0,0)
    Vertex();

    /// Create an unconnected vertex at (x,0,0)
    Vertex(real x);
    
    /// Create an unconnected vertex at (x,y,0)
    Vertex(real x, real y);

    /// Create an unconnected vertex at (x,y,z)
    Vertex(real x, real y, real z);

    /// Destructor
    ~Vertex();

    /// Clear vertex data
    void clear();
    
    ///--- Vertex data ---

    /// Return id of vertex
    int id() const;

    /// Return number of vertex neighbors
    int numVertexNeighbors() const;

    /// Return number of cell neighbors
    int numCellNeighbors() const;

    /// Return number of edge neighbors
    int numEdgeNeighbors() const;

    /// Return vertex neighbor number i
    Vertex& vertex(int i) const;

    /// Return cell neighbor number i
    Cell& cell(int i) const;

    /// Return edge neighbor number i
    Edge& edge(int i) const;

    /// Return parent vertex (null if no parent)
    Vertex* parent() const;

    /// Return child vertex (null if no child)
    Vertex* child() const;

    /// Return the mesh containing the vertex
    Mesh& mesh();
    
    /// Return the mesh containing the vertex (const version)
    const Mesh& mesh() const;

    /// Return vertex coordinate
    Point& coord();

    /// Return vertex coordinate
    Point coord() const;
    
    /// Return coordinate for midpoint on line to given vertex
    Point midpoint(const Vertex& n) const;

    /// Return distance to given vertex
    real dist(const Vertex& n) const;    

    /// Return distance to given point
    real dist(const Point& p) const;    
    
    /// Return distance to point with given coordinates
    real dist(real x, real y = 0.0, real z = 0.0) const;

    /// Check if given vertex is a neighbor
    bool neighbor(const Vertex& n) const;

    /// Comparison with another vertex
    bool operator==(const Vertex& vertex) const;

    /// Comparison with another vertex
    bool operator!=(const Vertex& vertex) const;

    /// Comparison based on the vertex id

    bool operator== (int id) const;
    bool operator<  (int id) const;
    bool operator<= (int id) const;
    bool operator>  (int id) const;
    bool operator>= (int id) const;

    friend bool operator== (int id, const Vertex& vertex);
    friend bool operator<  (int id, const Vertex& vertex);
    friend bool operator<= (int id, const Vertex& vertex);
    friend bool operator>  (int id, const Vertex& vertex);
    friend bool operator>= (int id, const Vertex& vertex);

    ///--- Output ---
   
    /// Display condensed vertex data
    friend LogStream& operator<<(LogStream& stream, const Vertex& vertex);
    
    /// Friends
    friend class Mesh;
    friend class MeshRefinement;
    friend class TriMeshRefinement;
    friend class TetMeshRefinement;
    friend class Triangle;
    friend class Tetrahedron;
    friend class MeshData;
    friend class MeshInit;
    friend class VertexIterator::VertexVertexIterator;
    friend class CellIterator::VertexCellIterator;	 
    friend class EdgeIterator::VertexEdgeIterator;	 
    
    // Boundary information
    std::set<int> nbids;

  private:

    // Specify global vertex number
    int setID(int id, Mesh& mesh);
    
    // Set the mesh pointer
    void setMesh(Mesh& mesh);

    // Set parent vertex
    void setParent(Vertex& parent);

    // Set child vertex
    void setChild(Vertex& child);

    // Remove parent vertex
    void removeParent(Vertex& parent);

    // Remove child vertex
    void removeChild();

    // Specify coordinate
    void set(real x, real y, real z);

    //--- Vertex data ---
    
    // The mesh containing this vertex
    Mesh* _mesh;

    // Global vertex number
    int _id;

    // Vertex coordinate
    Point p;

    // Connectivity
    PArray<Vertex*> nn;
    PArray<Cell*> nc;
    PArray<Edge*> ne;
    
    // Parent-child info
    Vertex* _parent;
    Vertex* _child;

  };
  
}

#endif
