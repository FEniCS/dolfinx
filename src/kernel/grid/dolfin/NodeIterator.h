// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NODE_ITERATOR_H
#define __NODE_ITERATOR_H

#include <dolfin/List.h>
#include <dolfin/ShortList.h>

namespace dolfin {
  
  class Grid;
  class Node;
  class Cell;
  class CellIterator;
  class GenericCell;

  typedef Node* NodePointer;
  
  class NodeIterator {
  public:
	 
	 NodeIterator(const Grid &grid);
	 NodeIterator(const Grid *grid);

	 NodeIterator(const Node &node);
	 NodeIterator(const NodeIterator &nodeIterator);
	 
	 NodeIterator(const Cell &cell);
	 NodeIterator(const CellIterator &cellIterator);
	 
	 ~NodeIterator();

	 operator NodePointer() const;
	
 	 NodeIterator& operator++();
	 bool end();
	 int index();
	 
	 Node& operator*() const;
	 Node* operator->() const;

  private:

	 // Base class for node iterators
	 class GenericNodeIterator {
	 public:
		
		virtual void operator++() = 0;
		virtual bool end() = 0;
		virtual int index() = 0;
		
		virtual Node& operator*() const = 0;
		virtual Node* operator->() const = 0;
		virtual Node* pointer() const = 0;
		
	 };
	 
	 // Iterator for the nodes in a grid
	 class GridNodeIterator : public GenericNodeIterator {
	 public:
		
		GridNodeIterator(const Grid &grid); 
		
		void operator++();
		bool end();
		int index();
		
		Node& operator*() const;
		Node* operator->() const;
		Node* pointer() const;

		List<Node>::Iterator node_iterator;
		List<Node>::Iterator at_end;
		
	 };

	 // Iterator for the node neighbors of a node
	 class NodeNodeIterator : public GenericNodeIterator {
	 public:

		NodeNodeIterator(const Node &node);
		void operator++();
		bool end();
		int index();

		Node& operator*() const;
		Node* operator->() const;
		Node* pointer() const;
		
	 private:

		ShortList<Node *>::Iterator node_iterator;
		
	 };
	 
	 // Iterator for the nodes in a cell
	 class CellNodeIterator : public GenericNodeIterator {
	 public:

		CellNodeIterator(const Cell &cell);
		void operator++();
		bool end();
		int index();

		Node& operator*() const;
		Node* operator->() const;
		Node* pointer() const;
		
	 private:

		ShortList<Node *>::Iterator node_iterator;

		GenericCell *genericCell;
		
	 };
	 
	 GenericNodeIterator *n;
	 
  };

}

#endif
