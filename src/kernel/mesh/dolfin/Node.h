// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NODE_HH
#define __NODE_HH

#include <dolfin/dolfin_log.h>
#include <dolfin/constants.h>
#include <dolfin/Array.h>
#include <dolfin/Point.h>
#include <dolfin/NodeIterator.h>
#include <dolfin/CellIterator.h>
#include <dolfin/EdgeIterator.h>

namespace dolfin{

  class GenericCell;
  class Cell;
  class Edge;
  class MeshInit;
  
  class Node{
  public:

    /// Create an unconnected node at (0,0,0)
    Node();

    /// Create an unconnected node at (x,0,0)
    Node(real x);
    
    /// Create an unconnected node at (x,y,0)
    Node(real x, real y);

    /// Create an unconnected node at (x,y,z)
    Node(real x, real y, real z);

    /// Destructor
    ~Node();

    /// Clear node data
    void clear();
    
    ///--- Node data ---

    /// Return id of node
    int id() const;

    /// Return number of node neighbors
    int noNodeNeighbors() const;

    /// Return number of cell neighbors
    int noCellNeighbors() const;

    /// Return number of edge neighbors
    int noEdgeNeighbors() const;

    /// Return node neighbor number i
    Node& node(int i) const;

    /// Return cell neighbor number i
    Cell& cell(int i) const;

    /// Return edge neighbor number i
    Edge& edge(int i) const;

    /// Return parent node (null if no parent)
    Node* parent() const;

    /// Return child node (null if no child)
    Node* child() const;

    /// Return node coordinate
    Point& coord();

    /// Return node coordinate
    Point coord() const;
    
    /// Return coordinate for midpoint on line to given node
    Point midpoint(const Node& n) const;

    /// Return distance to given node
    real dist(const Node& n) const;    

    /// Return distance to given point
    real dist(const Point& p) const;    
    
    /// Return distance to point with given coordinates
    real dist(real x, real y = 0.0, real z = 0.0) const;

    /// Check if given node is a neighbor
    bool neighbor(const Node& n) const;

    /// Comparison with another node
    bool operator==(const Node& node) const;

    /// Comparison with another node
    bool operator!=(const Node& node) const;

    /// Comparison based on the node id

    bool operator== (int id) const;
    bool operator<  (int id) const;
    bool operator<= (int id) const;
    bool operator>  (int id) const;
    bool operator>= (int id) const;

    friend bool operator== (int id, const Node& node);
    friend bool operator<  (int id, const Node& node);
    friend bool operator<= (int id, const Node& node);
    friend bool operator>  (int id, const Node& node);
    friend bool operator>= (int id, const Node& node);

    ///--- Output ---
   
    /// Display condensed node data
    friend LogStream& operator<<(LogStream& stream, const Node& node);
    
    /// Friends
    friend class Mesh;
    friend class MeshRefinement;
    friend class TriMeshRefinement;
    friend class TetMeshRefinement;
    friend class Triangle;
    friend class Tetrahedron;
    friend class MeshData;
    friend class MeshInit;
    friend class NodeIterator::NodeNodeIterator;
    friend class CellIterator::NodeCellIterator;	 
    friend class EdgeIterator::NodeEdgeIterator;	 
    
  private:

    // Specify global node number
    int setID(int id, Mesh& mesh);
    
    // Set the mesh pointer
    void setMesh(Mesh& mesh);

    // Set parent node
    void setParent(Node& parent);

    // Set child node
    void setChild(Node& child);

    // Remove parent node
    void removeParent(Node& parent);

    // Remove child node
    void removeChild();

    // Specify coordinate
    void set(real x, real y, real z);

    //--- Node data ---
    
    // The mesh containing this node
    Mesh* mesh;

    // Global node number
    int _id;

    // Node coordinate
    Point p;

    // Connectivity
    Array<Node*> nn;
    Array<Cell*> nc;
    Array<Edge*> ne;

    // Parent-child info
    Node* _parent;
    Node* _child;

  };
  
}

#endif
