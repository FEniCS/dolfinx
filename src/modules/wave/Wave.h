// Copyright (C) 2003 Johan Jansson
// Licensed under the GNU GPL Version 2.

#ifndef __WAVE_H
#define __WAVE_H

#include <dolfin/PDE.h>

namespace dolfin {
  
  class Wave : public PDE {
  public:
    
    Wave(Function& source,
	 Function& uprevious,
	 Function& wprevious) : PDE(2)
    {
      add(f,  source);
      add(up, uprevious);
      add(wp, wprevious);
    }
    
    real lhs(const ShapeFunction& u, const ShapeFunction& v)
    {
      return (1 / k * u*v + k *
	      (u.dx() * v.dx() + u.dy() * v.dy())) * dK;
    }
    
    real rhs(const ShapeFunction& v)
    {
      return (k * f * v + wp * v + 1 / k * up * v) * dK;
    }
    
  private:    
    ElementFunction f;          // Source term
    ElementFunction up; // Position value at left end-point
    ElementFunction wp; // Velocity value at left end-point
  };

}

#endif
