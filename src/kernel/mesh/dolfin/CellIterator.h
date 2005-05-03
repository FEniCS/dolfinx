// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#ifndef __CELL_ITERATOR_H
#define __CELL_ITERATOR_H

#include <dolfin/PArray.h>
#include <dolfin/Table.h>

namespace dolfin {

  class Mesh;
  class Cell;
  class Node;
  class NodeIterator;
  class Edge;
  class EdgeIterator;
  class Face;
  class FaceIterator;

  typedef Cell* CellPointer;
  
  class CellIterator {
  public:
	 
    CellIterator(const Mesh& mesh);
    CellIterator(const Mesh* mesh);

    CellIterator(const Node& node);
    CellIterator(const NodeIterator& nodeIterator);

    CellIterator(const Cell& cell);
    CellIterator(const CellIterator& cellIterator);

    CellIterator(const Edge& edge);
    CellIterator(const EdgeIterator& edgeIterator);

    CellIterator(const Face& face);
    CellIterator(const FaceIterator& faceIterator);
	 
    ~CellIterator();

    operator CellPointer() const;
	 
    CellIterator& operator++();
    bool end();
    bool last();
    int index();
	 
    Cell& operator*() const;
    Cell* operator->() const;
    bool  operator==(const CellIterator& c) const;
    bool  operator!=(const CellIterator& c) const;
    bool  operator==(const Cell& c) const;
    bool  operator!=(const Cell& c) const;
	 
    // Base class for cell iterators
    class GenericCellIterator {
    public:
		
      virtual void operator++() = 0;
      virtual bool end() = 0;
      virtual bool last() = 0;
      virtual int index() = 0;
		
      virtual Cell& operator*() const = 0;
      virtual Cell* operator->() const = 0;
      virtual Cell* pointer() const = 0;
		
    };
	 
    // Iterator for the cells in a mesh
    class MeshCellIterator : public GenericCellIterator {
    public:
		
      MeshCellIterator(const Mesh& mesh); 
		
      void operator++();
      bool end();
      bool last();
      int index();
		
      Cell& operator*() const;
      Cell* operator->() const;
      Cell* pointer() const;

      Table<Cell>::Iterator cell_iterator;
      Table<Cell>::Iterator at_end;
		
    };

    // Iterator for the cell neigbors of a node
    class NodeCellIterator : public GenericCellIterator {
    public:

      NodeCellIterator(const Node& node);
      void operator++();
      bool end();
      bool last();
      int index();

      Cell& operator*() const;
      Cell* operator->() const;
      Cell* pointer() const;

    private:

      PArray<Cell*>::Iterator cell_iterator;
		
    };

    // Iterator for the cell neigbors of a cell
    class CellCellIterator : public GenericCellIterator {
    public:

      CellCellIterator(const Cell& cell);
      void operator++();
      bool end();
      bool last();
      int index();

      Cell& operator*() const;
      Cell* operator->() const;
      Cell* pointer() const;

    private:

      PArray<Cell*>::Iterator cell_iterator;
		
    };

    // Iterator for the cell neigbors of an edge
    class EdgeCellIterator : public GenericCellIterator {
    public:

      EdgeCellIterator(const Edge& edge);
      void operator++();
      bool end();
      bool last();
      int index();

      Cell& operator*() const;
      Cell* operator->() const;
      Cell* pointer() const;

    private:

      PArray<Cell*>::Iterator cell_iterator;
		
    };

    // Iterator for the cell neigbors of a face
    class FaceCellIterator : public GenericCellIterator {
    public:

      FaceCellIterator(const Face& face);
      void operator++();
      bool end();
      bool last();
      int index();

      Cell& operator*() const;
      Cell* operator->() const;
      Cell* pointer() const;

    private:

      PArray<Cell*>::Iterator cell_iterator;
		
    };
	 
  private:
	 
    GenericCellIterator* c;
	 
  };

}

#endif
