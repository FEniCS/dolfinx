// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __EDGE_ITERATOR_H
#define __EDGE_ITERATOR_H

#include <dolfin/Array.h>
#include <dolfin/Table.h>

namespace dolfin {
  
  class Grid;
  class Node;
  class Cell;
  class Edge;
  class Face;
  class NodeIterator;
  class CellIterator;
  class FaceIterator;
  class GenericCell;

  typedef Edge* EdgePointer;
  
  class EdgeIterator {
  public:
	 
    EdgeIterator(const Grid& grid);
    EdgeIterator(const Grid* grid);

    EdgeIterator(const Node& node);
    EdgeIterator(const NodeIterator& nodeIterator);

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
		
      virtual Edge& operator*() const = 0;
      virtual Edge* operator->() const = 0;
      virtual Edge* pointer() const = 0;
		
    };
	 
    // Iterator for the edges in a grid
    class GridEdgeIterator : public GenericEdgeIterator {
    public:
		
      GridEdgeIterator(const Grid& grid); 
		
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

    // Iterator for the edges at a node 
    class NodeEdgeIterator : public GenericEdgeIterator {
    public:

      NodeEdgeIterator(const Node& node);
      void operator++();
      bool end();
      bool last();
      int index();

      Edge& operator*() const;
      Edge* operator->() const;
      Edge* pointer() const;
		
    private:

      Array<Edge*>::Iterator edge_iterator;
		
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

      Array<Edge*>::Iterator edge_iterator;

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

      Array<Edge*>::Iterator edge_iterator;

    };
	 
  private:

    GenericEdgeIterator* e;
	 
  };

}

#endif
