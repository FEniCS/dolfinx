// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __BOUNDARY_INIT_H
#define __BOUNDARY_INIT_H

namespace dolfin {

  class Grid;

  /// BoundaryInit implements the algorithm for computing the boundary
  /// of a given grid.

  class BoundaryInit {
  public:
    
    static void init(Grid& grid);
    
  private:

    static void clear(Grid& grid);

    static void initFaces(Grid& grid);
    static void initEdges(Grid& grid);
    static void initNodes(Grid& grid);

  };

}

#endif
