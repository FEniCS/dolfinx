// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __EDGE_HH
#define __EDGE_HH

namespace dolfin {

  class Node;
  class Point;
  class Grid;

  class Edge{
  public:
    
    Edge();
    Edge(Node *en1, Node *en2);
    ~Edge();
    
    void set(Node *en1, Node *en2);
    
    /// --- Accessor functions for stored data
  
      /// Get end node number i
      Node* node(int i);
      /// Get coordinates of end node number i
      Point coord(int i);
	
      /// --- Functions for mesh refinement
      void mark();
      void unmark();
      bool marked();
      
      /// --- Functions that require computation (every time!)
	  
      /// Compute and return the lenght of the edge
      real computeLength();
      /// Compute and return midpoint of the edge 
      Point computeMidpoint();
    
      /// Give access to the special functions below
      friend class Grid;
      friend class Node;
  
  protected:
  
  private:
	
      ShortList<Node*> end_nodes;

      bool marked_for_refinement;

  };

}

#endif
