// Copyright (C) 2002-2006 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2002
// Last changed: 2006-02-20

#ifndef __MESH_H
#define __MESH_H

#include <dolfin/dolfin_log.h>
#include <dolfin/Variable.h>
#include <dolfin/constants.h>
#include <dolfin/PList.h>
#include <dolfin/Point.h>
#include <dolfin/Vertex.h>
#include <dolfin/Cell.h>
#include <dolfin/Edge.h>
#include <dolfin/Face.h>
#include <dolfin/BoundaryData.h>
#include <dolfin/MeshData.h>

namespace dolfin
{

  class MeshData;

  /// A Mesh consists of Vertices, Cells, Edges, and Faces.
  /// The data of a Mesh is accessed through iterators:
  ///
  /// The Vertices of a Mesh is accessed through the class VertexIterator.
  /// The Cells of a Mesh is accessed through the class CellIterator.
  /// The Edges of a Mesh is accessed through the class EdgeIterator.
  /// The Faces of a Mesh is accessed through the class FaceIterator.
  ///
  /// A Cell can represent either a Triangle or a Tetrahedron depending
  /// on the type of the Mesh.
  
  class Mesh : public Variable
  {
  public:
    
    enum Type { triangles, tetrahedra };
    
    /// Create an empty mesh
    Mesh();

    /// Create mesh from given file
    Mesh(const char *filename);

    /// Copy constructor
    Mesh(const Mesh& mesh);

    /// Destructor
    ~Mesh();

    ///--- Basic functions

    /// Merge in another mesh. The two meshes are not connected by the
    /// operation.
    void merge(Mesh& mesh2);

    /// Compute connectivity
    void init();

    /// Clear mesh
    void clear();

    /// Return number of space dimensions of the mesh 
    int numSpaceDim() const;

    /// Return number of vertices in the mesh
    int numVertices() const;

    /// Return number of cells in the mesh
    int numCells() const;

    /// Return number of edges in the mesh
    int numEdges() const;

    /// Return number of faces in the mesh
    int numFaces() const;

    // Create a new vertex at given position
    Vertex& createVertex(Point p);
    Vertex& createVertex(real x, real y, real z);

    // Create a new cell from the given vertices
    Cell& createCell(int n0, int n1, int n2);
    Cell& createCell(int n0, int n1, int n2, int n3);
    Cell& createCell(Vertex& n0, Vertex& n1, Vertex& n2);
    Cell& createCell(Vertex& n0, Vertex& n1, Vertex& n2, Vertex& n3);

    // Create a new edge from the given vertices
    Edge& createEdge(int n0, int n1);
    Edge& createEdge(Vertex& n0, Vertex& n1);

    // Create a new face from the given edges
    Face& createFace(int e0, int e1, int e2);
    Face& createFace(Edge& e0, Edge& e1, Edge& e2);
    
    // Remove vertex, cell, edge, face (use with care)
    void remove(Vertex& vertex);
    void remove(Cell& cell);
    void remove(Edge& edge);
    void remove(Face& face);

    /// Return type of mesh
    Type type() const;

    /// Return given vertex (can also use a vertex iterator)
    Vertex& vertex(uint id);

    /// Return given cell (can also use a cell iterator)
    Cell& cell(uint id);

    /// Return given edge (can also use an edge iterator)
    Edge& edge(uint id);

    /// Return given face (can also use a face iterator)
    Face& face(uint id);

    /// Return boundary 
    Boundary boundary();

    ///--- Mesh refinement ---

    /// Refine mesh
    void refine();

    /// Refine uniformly (all cells marked)
    void refineUniformly();
    void refineUniformly(int i);

    /// Return parent mesh
    Mesh& parent();

    /// Return child mesh
    Mesh& child();

    /// Comparison of two meshs
    bool operator==(const Mesh& mesh) const;

    /// Comparison of two meshs
    bool operator!=(const Mesh& mesh) const;
    
    ///--- Output ---

    /// Display mesh data
    void disp() const;

    /// Display condensed mesh data
    friend LogStream& operator<< (LogStream& stream, const Mesh& mesh);
    
    /// Friends
    friend class GenericCell;
    friend class Edge;
    friend class XMLMesh;
    friend class MeshInit;
    friend class MeshRefinement;
    friend class TriMeshRefinement;
    friend class TetMeshRefinement;
    friend class MeshHierarchy;
    friend class Boundary;
    friend class BoundaryInit;
    friend class VertexIterator::MeshVertexIterator;
    friend class VertexIterator::BoundaryVertexIterator;
    friend class CellIterator::MeshCellIterator;
    friend class EdgeIterator::MeshEdgeIterator;
    friend class EdgeIterator::BoundaryEdgeIterator;
    friend class FaceIterator::MeshFaceIterator;
    friend class FaceIterator::BoundaryFaceIterator;
    
  private:

    // Create a new mesh as a child to this mesh
    Mesh& createChild();
    

    /// Swap data with given mesh
    void swap(Mesh& mesh);

    //--- Mesh data ---

    // Mesh data
    MeshData* md;

    // Boundary data
    BoundaryData* bd;

    // Parent mesh
    Mesh* _parent;

    // Child mesh
    Mesh* _child;
    
    // Mesh type
    Type _type;

  };
  
}

#endif
