// Copyright (C) 2003 Johan Jansson.
// Licensed under the GNU GPL Version 2.

#ifndef __WAVEVECTOR_H
#define __WAVEVECTOR_H

#include <dolfin/PDE.h>

namespace dolfin {
  
  class WaveVector : public PDE {
  public:
    
    WaveVector(Function::Vector& source,
	       Function::Vector& uprevious,
	       Function::Vector& wprevious) : PDE(2, 2)
    {
      add(f,  source);
      add(up, uprevious);
      add(wp, wprevious);
    }
    
    real lhs(ShapeFunction::Vector& u, ShapeFunction::Vector& v)
    {
      return
	(1 / k * u(0) * v(0) + k *
	 (u(0).ddx() * v(0).ddx() + u(0).ddy() * v(0).ddy())) * dK +
	(1 / k * u(1) * v(1) + k *
	 (u(1).ddx() * v(1).ddx() + u(1).ddy() * v(1).ddy())) * dK;
    }
    
    real rhs(ShapeFunction::Vector& v)
    {
	return
	  (k * f(0) * v(0) + wp(0) * v(0) + 1 / k * up(0) * v(0)) * dK +
	  (k * f(1) * v(1) + wp(1) * v(1) + 1 / k * up(1) * v(1)) * dK;
    }
    
  private:

    ElementFunction::Vector f;  // Source term
    ElementFunction::Vector up; // Position value at left end-point
    ElementFunction::Vector wp; // Velocity value at left end-point

  };

}

#endif
