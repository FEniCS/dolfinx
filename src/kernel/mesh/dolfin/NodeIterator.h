// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NODE_ITERATOR_H
#define __NODE_ITERATOR_H

#include <dolfin/Array.h>
#include <dolfin/List.h>
#include <dolfin/Table.h>

namespace dolfin {
  
  class Mesh;
  class Node;
  class Cell;
  class Boundary;
  class CellIterator;
  class GenericCell;
  
  typedef Node* NodePointer;
  
  class NodeIterator {
  public:
    
    NodeIterator(const Mesh& mesh);
    NodeIterator(const Mesh* mesh);

    NodeIterator(const Boundary& boundary);

    NodeIterator(const Node& node);
    NodeIterator(const NodeIterator& nodeIterator);
	 
    NodeIterator(const Cell& cell);
    NodeIterator(const CellIterator& cellIterator);
	 
    ~NodeIterator();

    operator NodePointer() const;
	
    NodeIterator& operator++();
    bool end();
    bool last();
    int index();
	 
    Node& operator*() const;
    Node* operator->() const;
    bool  operator==(const NodeIterator& n) const;
    bool  operator!=(const NodeIterator& n) const;
    bool  operator==(const Node& n) const;
    bool  operator!=(const Node& n) const;

    // Base class for node iterators
    class GenericNodeIterator {
    public:
		
      virtual void operator++() = 0;
      virtual bool end() = 0;
      virtual bool last() = 0;
      virtual int index() = 0;
		
      virtual Node& operator*() const = 0;
      virtual Node* operator->() const = 0;
      virtual Node* pointer() const = 0;
		
    };
	 
    // Iterator for the nodes in a mesh
    class MeshNodeIterator : public GenericNodeIterator {
    public:
		
      MeshNodeIterator(const Mesh& mesh); 
		
      void operator++();
      bool end();
      bool last();
      int index();
		
      Node& operator*() const;
      Node* operator->() const;
      Node* pointer() const;

      Table<Node>::Iterator node_iterator;
      Table<Node>::Iterator at_end;
		
    };

    // Iterator for the nodes on a boundary
    class BoundaryNodeIterator : public GenericNodeIterator {
    public:

      BoundaryNodeIterator(const Boundary& boundary);
      void operator++();
      bool end();
      bool last();
      int index();

      Node& operator*() const;
      Node* operator->() const;
      Node* pointer() const;
		
    private:

      List<Node*>::Iterator node_iterator;
      
    };

    // Iterator for the node neighbors of a node
    class NodeNodeIterator : public GenericNodeIterator {
    public:

      NodeNodeIterator(const Node& node);
      void operator++();
      bool end();
      bool last();
      int index();

      Node& operator*() const;
      Node* operator->() const;
      Node* pointer() const;
		
    private:

      Array<Node*>::Iterator node_iterator;
		
    };
	 
    // Iterator for the nodes in a cell
    class CellNodeIterator : public GenericNodeIterator {
    public:

      CellNodeIterator(const Cell& cell);
      void operator++();
      bool end();
      bool last();
      int index();

      Node& operator*() const;
      Node* operator->() const;
      Node* pointer() const;
		
    private:

      Array<Node*>::Iterator node_iterator;
		
    };
	 
  private:

    GenericNodeIterator* n;
	 
  };

}

#endif
