// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// Map from reference element (0,0,0) - (1,0,0) - (0,1,0) - (0,0,1)
// to a given tetrahedron in in R^3.

#ifndef __P1_TET_MAP_H
#define __P1_TET_MAP_H

#include <dolfin/Map.h>

namespace dolfin {
  
  class P1TetMap : public Map {
  public:
    
    P1TetMap();
    
    const FunctionSpace::ElementFunction dx(const FunctionSpace::ShapeFunction &v) const;
    const FunctionSpace::ElementFunction dy(const FunctionSpace::ShapeFunction &v) const;
    const FunctionSpace::ElementFunction dz(const FunctionSpace::ShapeFunction &v) const;
    const FunctionSpace::ElementFunction dt(const FunctionSpace::ShapeFunction &v) const;

    void update(const Cell &cell);
    
  };

}

#endif
