// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __GRID_INIT_H
#define __GRID_INIT_H

namespace dolfin{

  class Grid;

  /// GridInit implements the algorithm for computing the neighbour
  /// information (connections) in a grid.
  ///
  /// The trick is to compute the connections in the correct order, as
  /// indicated in GridInit.h, to obtain an O(n) algorithm.
  
  class GridInit {
  public:
    
    static void init(Grid& grid);
    
  private:
    
    static void clear            (Grid& grid);
    
    static void initEdges        (Grid& grid);
    static void initConnectivity (Grid& grid);
    static void initFaces        (Grid& grid);
    
    static void initNodeCell     (Grid& grid);
    static void initCellCell     (Grid& grid);
    static void initNodeEdge     (Grid& grid);
    static void initNodeNode     (Grid& grid);
    
  };

}

#endif
