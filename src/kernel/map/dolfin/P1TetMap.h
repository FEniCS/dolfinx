// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// Map from the reference cell defined by the nodes
// (0,0,0) - (1,0,0) - (0,1,0) - (0,0,1) to a given tetrahedron in 3D.

#ifndef __P1_TET_MAP_H
#define __P1_TET_MAP_H

#include <dolfin/Map.h>

namespace dolfin {
  
  class P1TetMap : public Map {
  public:
    
    P1TetMap();

    void update(const Cell &cell);
    void update(const Edge& bnd_edge);
    void update(const Face& bnd_face);
    
    const FunctionSpace::ElementFunction ddx(const FunctionSpace::ShapeFunction &v) const;
    const FunctionSpace::ElementFunction ddy(const FunctionSpace::ShapeFunction &v) const;
    const FunctionSpace::ElementFunction ddz(const FunctionSpace::ShapeFunction &v) const;
    const FunctionSpace::ElementFunction ddt(const FunctionSpace::ShapeFunction &v) const;
    
  };

}

#endif
