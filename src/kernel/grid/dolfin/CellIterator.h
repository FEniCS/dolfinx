// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __CELL_ITERATOR_H
#define __CELL_ITERATOR_H

#include <dolfin/List.h>
#include <dolfin/ShortList.h>

namespace dolfin {

  class Grid;
  class Cell;
  class Node;
  class NodeIterator;

  typedef Cell* CellPointer;
  
  class CellIterator {
  public:
	 
	 CellIterator(Grid &grid);
	 CellIterator(Grid *grid);

	 CellIterator(Cell &cell);
	 CellIterator(CellIterator &cellIterator);
	 
	 CellIterator(Node &node);
	 CellIterator(NodeIterator &nodeIterator);
	 
	 ~CellIterator();

	 operator Cell&() const;
	 operator CellPointer() const;
	 
	 CellIterator& operator++();
	 bool end();
	 int index();
	 
	 Cell& operator*() const;
	 Cell* operator->() const;

  private:

	 // Base class for cell iterators
	 class GenericCellIterator {
	 public:
		
		virtual void operator++() = 0;
		virtual bool end() = 0;
		virtual int index() = 0;
		
		virtual Cell& operator*() const = 0;
		virtual Cell* operator->() const = 0;
		virtual Cell* pointer() const = 0;
		
	 };
	 
	 // Iterator for the cells in a grid
	 class GridCellIterator : public GenericCellIterator {
	 public:
		
		GridCellIterator(Grid &grid); 
		
		void operator++();
		bool end();
		int index();
		
		Cell& operator*() const;
		Cell* operator->() const;
		Cell* pointer() const;

		List<Cell>::Iterator cell_iterator;
		List<Cell>::Iterator at_end;
		
	 };

	 // Iterator for the cell neigbors of a cell
	 class CellCellIterator : public GenericCellIterator {
	 public:

		CellCellIterator(Cell &cell);
		void operator++();
		bool end();
		int index();

		Cell& operator*() const;
		Cell* operator->() const;
		Cell* pointer() const;

	 private:

		ShortList<Cell *>::Iterator cell_iterator;
		
	 };
	 
	 // Iterator for the cell neigbors of a node
	 class NodeCellIterator : public GenericCellIterator {
	 public:

		NodeCellIterator(Node &node);
		void operator++();
		bool end();
		int index();

		Cell& operator*() const;
		Cell* operator->() const;
		Cell* pointer() const;

	 private:

		ShortList<Cell *>::Iterator cell_iterator;
		
	 };

	 GenericCellIterator *c;
	 
  };

}

#endif
