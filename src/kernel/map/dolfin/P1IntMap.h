// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// Map from the reference cell defined by the nodes
// (0,0,0) - (1,0,0) to a given line in R^3.

#ifndef __P1_INT_MAP_H
#define __P1_INT_MAP_H

#include <dolfin/Map.h>

namespace dolfin {
  
  class P1IntMap : public Map {
  public:
    
    P1IntMap();
    
    const FunctionSpace::ElementFunction ddx(const FunctionSpace::ShapeFunction &v) const;
    const FunctionSpace::ElementFunction ddy(const FunctionSpace::ShapeFunction &v) const;
    const FunctionSpace::ElementFunction ddz(const FunctionSpace::ShapeFunction &v) const;
    const FunctionSpace::ElementFunction ddt(const FunctionSpace::ShapeFunction &v) const;
    
    void update(const Cell &cell);
    
  };

}

#endif
