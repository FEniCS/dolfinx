// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __CELL_ITERATOR_H
#define __CELL_ITERATOR_H

#include <dolfin/Array.h>
#include <dolfin/Table.h>

namespace dolfin {

  class Grid;
  class Cell;
  class Node;
  class NodeIterator;

  typedef Cell* CellPointer;
  
  class CellIterator {
  public:
	 
    CellIterator(const Grid& grid);
    CellIterator(const Grid* grid);

    CellIterator(const Node& node);
    CellIterator(const NodeIterator& nodeIterator);

    CellIterator(const Cell& cell);
    CellIterator(const CellIterator& cellIterator);
	 
    ~CellIterator();

    operator Cell&() const;
    operator CellPointer() const;
	 
    CellIterator& operator++();
    bool end();
    bool last();
    int index();
	 
    Cell& operator*() const;
    Cell* operator->() const;
    bool  operator==(const CellIterator& c) const;
    bool  operator!=(const CellIterator& c) const;
	 
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
	 
    // Iterator for the cells in a grid
    class GridCellIterator : public GenericCellIterator {
    public:
		
      GridCellIterator(const Grid& grid); 
		
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

      Array<Cell*>::Iterator cell_iterator;
		
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

      Array<Cell*>::Iterator cell_iterator;
		
    };
	 
  private:
	 
    GenericCellIterator* c;
	 
  };

}

#endif
