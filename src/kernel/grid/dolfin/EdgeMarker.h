// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __EDGE_MARKER_H
#define __EDGE_MARKER_H

namespace dolfin {

  enum EdgeMark { marked, unmarked };
  
  /// Edge marker
  class EdgeMarker {
  public:

    /// Create an empty marker
    EdgeMarker() {
      mark = unmarked;
    }

    /// The mark of the edge
    EdgeMark mark;
    
  };

}

#endif
