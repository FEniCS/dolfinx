// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __EDGE_REF_DATA_H
#define __EDGE_REF_DATA_H

namespace dolfin {

  class Cell;

  /// Edge refinement data
  class EdgeRefData {
  public:
    
    /// Mark edge by given cell
    void mark(Cell& cell);

    /// Check if edge has been marked
    bool marked() const;

    /// Check if edge has been marked by given cell
    bool marked(Cell& cell);

  private:

    /// Cells that have marked the edge
    List<Cell*> cells;
    
  };

}

#endif
