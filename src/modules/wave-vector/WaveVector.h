// Copyright (C) 2002 Johan Hoffman and Anders Logg.
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
	 (u(0).dx() * v(0).dx() + u(0).dy() * v(0).dy())) * dK +
	(1 / k * u(1) * v(1) + k *
	 (u(1).dx() * v(1).dx() + u(1).dy() * v(1).dy())) * dK;
    }
    
    real rhs(ShapeFunction::Vector& v)
    {
	return
	  (k * f(0) * v(0) + wp(0) * v(0) + 1 / k * up(0) * v(0)) * dK +
	  (k * f(1) * v(1) + wp(1) * v(1) + 1 / k * up(1) * v(1)) * dK;
    }
    
  private:    
    ElementFunction::Vector f;          // Source term
    ElementFunction::Vector up; // Position value at left end-point
    ElementFunction::Vector wp; // Velocity value at left end-point
  };

}

#endif
