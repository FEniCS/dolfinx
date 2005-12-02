// Copyright (C) 2003-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2005.
//
// First added:  2003
// Last changed: 2005-12-01

#ifndef __EDGE_ITERATOR_H
#define __EDGE_ITERATOR_H

#include <dolfin/PArray.h>
#include <dolfin/PList.h>
#include <dolfin/Table.h>

namespace dolfin
{
  
  class Mesh;
  class Vertex;
  class Cell;
  class Edge;
  class Face;
  class Boundary;
  class VertexIterator;
  class CellIterator;
  class FaceIterator;
  class GenericCell;

  typedef Edge* EdgePointer;
  
  class EdgeIterator
  {
  public:
	 
    EdgeIterator(const Mesh& mesh);
    EdgeIterator(const Mesh* mesh);

    EdgeIterator(const Boundary& boundary);
    EdgeIterator(const Boundary* boundary);

    EdgeIterator(const Vertex& vertex);
    EdgeIterator(const VertexIterator& vertexIterator);

    EdgeIterator(const Cell& cell);
    EdgeIterator(const CellIterator& cellIterator);

    EdgeIterator(const Face& face);
    EdgeIterator(const FaceIterator& faceIterator);
	 
    ~EdgeIterator();

    operator EdgePointer() const;
	
    EdgeIterator& operator++();
    bool end();
    bool last();
    int index();
	 
    Edge& operator*() const;
    Edge* operator->() const;
    bool  operator==(const EdgeIterator& n) const;
    bool  operator!=(const EdgeIterator& n) const;
	 
    // Base class for edge iterators
    class GenericEdgeIterator {
    public:
		
      virtual void operator++() = 0;
      virtual bool end() = 0;
      virtual bool last() = 0;
      virtual int index() = 0;

      virtual ~GenericEdgeIterator() {};
		
      virtual Edge& operator*() const = 0;
      virtual Edge* operator->() const = 0;
      virtual Edge* pointer() const = 0;
		
    };
	 
    // Iterator for the edges in a mesh
    class MeshEdgeIterator : public GenericEdgeIterator {
    public:
		
      MeshEdgeIterator(const Mesh& mesh); 
		
      void operator++();
      bool end();
      bool last();
      int index();
		
      Edge& operator*() const;
      Edge* operator->() const;
      Edge* pointer() const;

      Table<Edge>::Iterator edge_iterator;
      Table<Edge>::Iterator at_end;
		
    };

    // Iterator for the edges on a boundary
    class BoundaryEdgeIterator : public GenericEdgeIterator {
    public:

      BoundaryEdgeIterator(const Boundary& boundary);
      void operator++();
      bool end();
      bool last();
      int index();

      Edge& operator*() const;
      Edge* operator->() const;
      Edge* pointer() const;
		
    private:

      PList<Edge*>::Iterator edge_iterator;
      
    };

    // Iterator for the edges at a vertex 
    class VertexEdgeIterator : public GenericEdgeIterator {
    public:

      VertexEdgeIterator(const Vertex& vertex);
      void operator++();
      bool end();
      bool last();
      int index();

      Edge& operator*() const;
      Edge* operator->() const;
      Edge* pointer() const;
		
    private:

      PArray<Edge*>::Iterator edge_iterator;
		
    };
	 
    // Iterator for the edges in a cell
    class CellEdgeIterator : public GenericEdgeIterator {
    public:

      CellEdgeIterator(const Cell& cell);
      void operator++();
      bool end();
      bool last();
      int index();

      Edge& operator*() const;
      Edge* operator->() const;
      Edge* pointer() const;
		
    private:

      PArray<Edge*>::Iterator edge_iterator;

    };

    // Iterator for the edges in a face
    class FaceEdgeIterator : public GenericEdgeIterator {
    public:

      FaceEdgeIterator(const Face& face);
      void operator++();
      bool end();
      bool last();
      int index();

      Edge& operator*() const;
      Edge* operator->() const;
      Edge* pointer() const;
		
    private:

      PArray<Edge*>::Iterator edge_iterator;

    };
	 
  private:

    GenericEdgeIterator* e;
	 
  };

}

#endif
