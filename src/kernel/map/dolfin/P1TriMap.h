// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// Map from reference element (0,0) - (1,0) - (0,1)
// to a given triangle in R^2 (i.e. z = 0).

#ifndef __P1_TRI_MAP_H
#define __P1_TRI_MAP_H

#include <dolfin/Map.h>

namespace dolfin {
  
  class P1TriMap : public Map {
  public:
    
    P1TriMap();
    
    const FunctionSpace::ElementFunction dx(const FunctionSpace::ShapeFunction &v) const;
    const FunctionSpace::ElementFunction dy(const FunctionSpace::ShapeFunction &v) const;
    const FunctionSpace::ElementFunction dz(const FunctionSpace::ShapeFunction &v) const;
    const FunctionSpace::ElementFunction dt(const FunctionSpace::ShapeFunction &v) const;
    
    void update(const Cell &cell);
    
  };

}

#endif
