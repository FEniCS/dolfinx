// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __TRIANGLE_H
#define __TRIANGLE_H

#include <dolfin/Cell.h>
#include <dolfin/GenericCell.h>

namespace dolfin {

  class Node;
  class Grid;
  
  class Triangle : public GenericCell {
  public:
	 
	 Triangle();
	 ~Triangle();

	 int noNodes();
	 int noEdges();
	 int noFaces();
	 int noBoundaries();
	 
	 Cell::Type type();
	 
	 void set(Node *n0, Node *n1, Node *n2);
	 
	 void Set(Node *n1, Node *n2, Node *n3, int material);
	 
	 int GetSize ();
	 Node* GetNode (int node);
	 
	 real ComputeVolume       (Grid *grid);
	 real ComputeCircumRadius (Grid *grid);
	 real ComputeCircumRadius (Grid *grid, real volume);

	 /// Output
	 friend std::ostream& operator << (std::ostream& output, const Triangle& t);
	 
	 /// Friends
	 friend class Grid;
	 friend class Node;
	 friend class GridData;
	 
  protected:
	 
	 void CountCell            (Node *node_list);
	 void AddCell              (Node *node_list, int *current, int thiscell);
	 void AddNodes             (int exclude_node, int *new_nodes, int *pos);
	 void ComputeCellNeighbors (Node *node_list, int thiscell);
	 
  private:

	 Node* getNode(int i);
	 bool  neighbor(ShortList<Node *> &cn, Cell &cell);
	 
	 Node* nodes[3];

  };

}

#endif
