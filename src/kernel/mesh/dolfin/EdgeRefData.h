// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#ifndef __EDGE_REF_DATA_H
#define __EDGE_REF_DATA_H

namespace dolfin
{

  class Cell;

  /// Edge refinement data
  class EdgeRefData
  {
  public:
    
    /// Mark edge by given cell
    void mark(Cell& cell);

    /// Check if edge has been marked
    bool marked() const;

    /// Check if edge has been marked by given cell
    bool marked(Cell& cell);

    /// Clear marks
    void clear();

  private:

    /// Cells that have marked the edge
    PList<Cell*> cells;
    
  };

}

#endif
