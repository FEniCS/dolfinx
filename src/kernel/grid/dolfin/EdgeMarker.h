// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __EDGE_MARKER_H
#define __EDGE_MARKER_H

namespace dolfin {

  /// Edge marker
  class EdgeMarker {
  public:

    /// Create an empty marker
    EdgeMarker() {
      cellcount = 0;
    }

    /// Number of cells this edge has been marked by
    int cellcount;
    
  };

}

#endif
