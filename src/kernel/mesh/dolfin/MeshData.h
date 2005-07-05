// Copyright (C) 2002-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2002
// Last changed: 2005
//
// A couple of comments:
//
//   - Is the check in createNode() really necessary?

#ifndef __MESH_DATA_H
#define __MESH_DATA_H

#include <dolfin/Table.h>
#include <dolfin/Node.h>
#include <dolfin/Cell.h>
#include <dolfin/Edge.h>
#include <dolfin/Face.h>

namespace dolfin {

  /// MeshData is a container for mesh data.
  ///
  /// A Table (block-linked list) is used to store the mesh data:
  ///
  ///   a table of all nodes (n)
  ///   a table of all cells (c)
  ///   a table of all edges (e)
  ///   a table of all faces (f)
  ///
  /// Connectivity is stored locally. With four different geometric
  /// objects (n, c, e, f), 16 different combinations are
  /// possible. The combinations marked with an (x) are computed and
  /// stored:
  ///
  ///   (x) n-n (the node neighbors of a node)  [5, from 4]
  ///   (x) n-c (the cell neighbors of a node)  [1, from 0]
  ///   (x) n-e (the edge neighbors of a node)  [4, from 3]
  ///       n-f (the face neighbors of a node)
  ///
  ///   (x) c-n (the nodes within a cell)       [0]
  ///   (x) c-c (the cell neighbors of a cell)  [2, from 0 and 1]
  ///   (x) c-e (the edges within a cell)       [3, from 0 and 2]
  ///   (x) c-f (the faces within a cell)       [6, from 0 and 2]
  ///
  ///   (x) e-n (the nodes within an edge)      [3, from 0 and 2]
  ///   (x) e-c (the cell neighbors of an edge) [7, from 3
  ///       e-e (the edge neighbors of an edge)
  ///       e-f (the face neighbors of an edge)
  ///
  ///       f-n (the nodes within a face)
  ///   (x) f-c (the cell neighbors of a face)  [8, from 6]
  ///   (x) f-e (the edges within a face)       [6, from 1 and 3]
  ///       f-f (the face neighbors of a face)
  ///
  /// The numbers within brackets indicate in which order the
  /// connectivity is computed. A [0] indicates that the information
  /// is known a priori.
  ///
  /// Clarification:
  ///
  /// - Two nodes are neighbors if they are part of the same edge.
  ///   Each node is a neighbor to itself.
  /// - Two cells are neighbors if they share a common edge.
  ///   Each cell is a neighbor to itself.

  class MeshData {
  public:
    
    /// Create an empty set of mesh data
    MeshData(Mesh& mesh);

    /// Destructor
    ~MeshData();

    /// Clear all data
    void clear();

    Node& createNode(Point p);
    Node& createNode(real x, real y, real z);
    
    Cell& createCell(int n0, int n1, int n2);
    Cell& createCell(int n0, int n1, int n2, int n3);
    Cell& createCell(Node& n0, Node& n1, Node& n2);
    Cell& createCell(Node& n0, Node& n1, Node& n2, Node& n3);

    Edge& createEdge(int n0, int n1);
    Edge& createEdge(Node& n0, Node& n1);

    Face& createFace(int e0, int e1, int e2);
    Face& createFace(Edge& e0, Edge& e1, Edge& e2);
    
    Node& node(int id);
    Cell& cell(int id);
    Edge& edge(int id);
    Face& face(int id);

    void remove(Node& node);
    void remove(Cell& cell);
    void remove(Edge& edge);
    void remove(Face& face);
    
    int noNodes() const;
    int noCells() const;
    int noEdges() const;
    int noFaces() const;

    // Friends
    friend class Mesh;
    friend class MeshInit;
    friend class NodeIterator::MeshNodeIterator;
    friend class CellIterator::MeshCellIterator;
    friend class EdgeIterator::MeshEdgeIterator;
    friend class FaceIterator::MeshFaceIterator;
    
  private:
    
    // Change the mesh pointer
    void setMesh(Mesh& mesh);

    // The mesh
    Mesh* mesh;

    // Table of all nodes within the mesh
    Table<Node> nodes;

    // Table of all cells within the mesh
    Table<Cell> cells;

    // Table of all edges within the mesh
    Table<Edge> edges;

    // Table of all faces within the mesh
    Table<Face> faces;
   
  };
  
}
  
#endif
