// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __BOUNDARY_DATA_H
#define __BOUNDARY_DATA_H

#include <dolfin/List.h>

namespace dolfin {

  class Node;
  class Edge;
  class Face;
  class Grid;

  class BoundaryData {
  public:
    
    // Create an empty set of boundary data
    BoundaryData(Grid* grid);

    /// Destructor
    ~BoundaryData();

    /// Clear all data
    void clear();
    
    /// Add node to the boundary
    void add(Node* node);

    /// Add edge to the boundary
    void add(Edge* edge);

    /// Add face to the boundary
    void add(Face* face);

    /// Check if the boundary is empty
    bool empty();

  private:

    // The grid
    Grid* grid;
    
    // A list of all nodes on the boundary
    List<Node*> nodes;

    // A list of all edges on the boundary
    List<Edge*> edges;

    // A list of all faces on the boundary
    List<Face*> faces;

  };

}

#endif
