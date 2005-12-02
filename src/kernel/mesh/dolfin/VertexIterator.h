// Copyright (C) 2002-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2005.
//
// First added:  2002
// Last changed: 2005-12-01

#ifndef __VERTEX_ITERATOR_H
#define __VERTEX_ITERATOR_H

#include <dolfin/PArray.h>
#include <dolfin/PList.h>
#include <dolfin/Table.h>

namespace dolfin
{
  
  class Mesh;
  class Vertex;
  class Cell;
  class Boundary;
  class CellIterator;
  class GenericCell;
  
  typedef Vertex* VertexPointer;
  
  class VertexIterator
  {
  public:
    
    VertexIterator(const Mesh& mesh);
    VertexIterator(const Mesh* mesh);

    VertexIterator(const Boundary& boundary);

    VertexIterator(const Vertex& vertex);
    VertexIterator(const VertexIterator& vertexIterator);
	 
    VertexIterator(const Cell& cell);
    VertexIterator(const CellIterator& cellIterator);
	 
    ~VertexIterator();

    operator VertexPointer() const;
	
    VertexIterator& operator++();
    bool end();
    bool last();
    int index();
	 
    Vertex& operator*() const;
    Vertex* operator->() const;
    bool  operator==(const VertexIterator& n) const;
    bool  operator!=(const VertexIterator& n) const;
    bool  operator==(const Vertex& n) const;
    bool  operator!=(const Vertex& n) const;

    // Base class for vertex iterators
    class GenericVertexIterator {
    public:
		
      virtual void operator++() = 0;
      virtual bool end() = 0;
      virtual bool last() = 0;
      virtual int index() = 0;

      virtual ~GenericVertexIterator() {};
		
      virtual Vertex& operator*() const = 0;
      virtual Vertex* operator->() const = 0;
      virtual Vertex* pointer() const = 0;
		
    };
	 
    // Iterator for the vertices in a mesh
    class MeshVertexIterator : public GenericVertexIterator {
    public:
		
      MeshVertexIterator(const Mesh& mesh); 
		
      void operator++();
      bool end();
      bool last();
      int index();
		
      Vertex& operator*() const;
      Vertex* operator->() const;
      Vertex* pointer() const;

      Table<Vertex>::Iterator vertex_iterator;
      Table<Vertex>::Iterator at_end;
		
    };

    // Iterator for the vertices on a boundary
    class BoundaryVertexIterator : public GenericVertexIterator {
    public:

      BoundaryVertexIterator(const Boundary& boundary);
      void operator++();
      bool end();
      bool last();
      int index();

      Vertex& operator*() const;
      Vertex* operator->() const;
      Vertex* pointer() const;
		
    private:

      PList<Vertex*>::Iterator vertex_iterator;
      
    };

    // Iterator for the vertex neighbors of a vertex
    class VertexVertexIterator : public GenericVertexIterator {
    public:

      VertexVertexIterator(const Vertex& vertex);
      void operator++();
      bool end();
      bool last();
      int index();

      Vertex& operator*() const;
      Vertex* operator->() const;
      Vertex* pointer() const;
		
    private:

      PArray<Vertex*>::Iterator vertex_iterator;
		
    };

    // Iterator for the vertices in a cell
    class CellVertexIterator : public GenericVertexIterator {
    public:

      CellVertexIterator(const Cell& cell);
      void operator++();
      bool end();
      bool last();
      int index();

      Vertex& operator*() const;
      Vertex* operator->() const;
      Vertex* pointer() const;
		
    private:

      PArray<Vertex*>::Iterator vertex_iterator;
		
    };
	 
  private:

    GenericVertexIterator* n;
	 
  };

}

#endif
