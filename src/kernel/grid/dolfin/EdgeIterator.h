// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __EDGE_ITERATOR_H
#define __EDGE_ITERATOR_H

#include <dolfin/List.h>
#include <dolfin/ShortList.h>

namespace dolfin {
  
  class Grid;
  class Node;
  class Cell;
  class Edge;
  class CellIterator;
  class NodeIterator;
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
	 
	 EdgeIterator(const Edge& edge);
	 EdgeIterator(const EdgeIterator& edgeIterator);
	 
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
	 
	 // Iterator for the nodes in a grid
	 class GridEdgeIterator : public GenericEdgeIterator {
	 public:
		
		GridEdgeIterator(const Grid &grid); 
		
		void operator++();
		bool end();
		bool last();
		int index();
		
		Edge& operator*() const;
		Edge* operator->() const;
		Edge* pointer() const;

		List<Edge>::Iterator edge_iterator;
		List<Edge>::Iterator at_end;
		
	 };

	 
	 // Iterator for the edges in a cell
	 class CellEdgeIterator : public GenericEdgeIterator {
	 public:

		CellEdgeIterator(const Cell &cell);
		void operator++();
		bool end();
		bool last();
		int index();

		Edge& operator*() const;
		Edge* operator->() const;
		Edge* pointer() const;
		
	 private:

		ShortList<Edge *>::Iterator edge_iterator;

		GenericCell *genericCell;
		
	 };
	 

	 // Iterator for the edges at a node 
	 class NodeEdgeIterator : public GenericEdgeIterator {
	 public:

		NodeEdgeIterator(const Node &node);
		void operator++();
		bool end();
		bool last();
		int index();

		Edge& operator*() const;
		Edge* operator->() const;
		Edge* pointer() const;
		
	 private:

		ShortList<Edge *>::Iterator edge_iterator;

		GenericCell *genericCell;
		
	 };
	 
  private:

	 GenericEdgeIterator *n;
	 
  };

}

#endif
