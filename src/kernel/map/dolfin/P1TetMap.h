// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// Map from the reference cell defined by the nodes
// (0,0,0) - (1,0,0) - (0,1,0) - (0,0,1) to a given tetrahedron in 3D.

#ifndef __P1_TET_MAP_H
#define __P1_TET_MAP_H

#include <dolfin/Map.h>

namespace dolfin {
  
  class P1TetMap : public Map
  {
  public:
    
    P1TetMap();

    Point operator() (const Point& p) const;
    Point operator() (const Point& p, unsigned int boundary) const;

    void update(const Cell& cell);
    void update(const Face& face);
    
    const FunctionSpace::ElementFunction ddx(const FunctionSpace::ShapeFunction &v) const;
    const FunctionSpace::ElementFunction ddy(const FunctionSpace::ShapeFunction &v) const;
    const FunctionSpace::ElementFunction ddz(const FunctionSpace::ShapeFunction &v) const;
    const FunctionSpace::ElementFunction ddt(const FunctionSpace::ShapeFunction &v) const;

  private:

    // FIXME: Temporary hack until we sort things out with the
    // FIXME: local numbering of nodes, cells, edges, and faces.
    unsigned int faceNumber(const Face& face, const Cell& cell) const;
    
  };

}

#endif
