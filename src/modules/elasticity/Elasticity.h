// Copyright (C) 2003 Fredrik Bengzon and Johan Jansson.
// Licensed under the GNU GPL Version 2.

#ifndef __ELASTICITY_H
#define __ELASTICITY_H

#include <dolfin/PDE.h>

namespace dolfin {
  
  class Elasticity : public PDE {
  public:
    
    Elasticity(Function::Vector& source,
	       Function::Vector& uprevious,
	       Function::Vector& wprevious) : PDE(3, 3), f(3), up(3), wp(3)
    {
      add(f,  source);
      add(up, uprevious);
      add(wp, wprevious);
    }
    
    real lhs(ShapeFunction::Vector& u, ShapeFunction::Vector& v)
    {
      real b = 0.05;
      real E = 20.0;
      real nu = 0.3;


      real lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
      real mu = E / (2 * (1 + nu));

      ElementFunction
	epsilon00, epsilon01, epsilon02,
	epsilon10, epsilon11, epsilon12,
	epsilon20, epsilon21, epsilon22;

      ElementFunction
	epsilonv00, epsilonv01, epsilonv02,
	epsilonv10, epsilonv11, epsilonv12,
	epsilonv20, epsilonv21, epsilonv22;

      ElementFunction
	sigma00, sigma01, sigma02,
	sigma10, sigma11, sigma12,
	sigma20, sigma21, sigma22;

      epsilon00 = u(0).ddx();
      epsilon01 = 0.5 * (u(0).ddy() + u(1).ddx());
      epsilon02 = 0.5 * (u(0).ddz() + u(2).ddx());
      epsilon10 = 0.5 * (u(1).ddx() + u(0).ddy());
      epsilon11 = u(1).ddy();
      epsilon12 = 0.5 * (u(1).ddz() + u(2).ddy());
      epsilon20 = 0.5 * (u(2).ddx() + u(0).ddz());
      epsilon21 = 0.5 * (u(2).ddy() + u(1).ddz());
      epsilon22 = u(2).ddz();

      epsilonv00 = v(0).ddx();
      epsilonv01 = 0.5 * (v(0).ddy() + v(1).ddx());
      epsilonv02 = 0.5 * (v(0).ddz() + v(2).ddx());
      epsilonv10 = 0.5 * (v(1).ddx() + v(0).ddy());
      epsilonv11 = v(1).ddy();
      epsilonv12 = 0.5 * (v(1).ddz() + v(2).ddy());
      epsilonv20 = 0.5 * (v(2).ddx() + v(0).ddz());
      epsilonv21 = 0.5 * (v(2).ddy() + v(1).ddz());
      epsilonv22 = v(2).ddz();

      sigma00 = lambda * (u(0).ddx() + u(1).ddy() + u(2).ddz()) +
	2 * mu * epsilon00;
      sigma01 = 2 * mu * epsilon01;
      sigma02 = 2 * mu * epsilon02;
      sigma10 = 2 * mu * epsilon10;
      sigma11 = lambda * (u(0).ddx() + u(1).ddy() + u(2).ddz()) +
	2 * mu * epsilon11;
      sigma12 = 2 * mu * epsilon12;
      sigma20 = 2 * mu * epsilon20;
      sigma21 = 2 * mu * epsilon21;
      sigma22 = lambda * (u(0).ddx() + u(1).ddy() + u(2).ddz()) +
	2 * mu * epsilon22;

      return
	((u(0) * v(0) + u(1) * v(1) + u(2) * v(2)) + k * k *
	 (sigma00 * epsilonv00 + sigma01 * epsilonv01 + sigma02 * epsilonv02 +
	  sigma10 * epsilonv10 + sigma11 * epsilonv11 + sigma12 * epsilonv12 +
	  sigma20 * epsilonv20 + sigma21 * epsilonv21 + sigma22 * epsilonv22)) * dx;
    }
    
    real rhs(ShapeFunction::Vector& v)
    {
      return ((up(0) * v(0) + up(1) * v(1) + up(2) * v(2)) +
	      k * (wp(0) * v(0) + wp(1) * v(1) + wp(2) * v(2)) +
	      k * k * (f(0) * v(0) + f(1) * v(1) + f(2) * v(2))) * dx;
    }
    
  private:    
    ElementFunction::Vector f; // Source term
    ElementFunction::Vector up;        // Displacement value at left end-point
    ElementFunction::Vector wp;        // Velocity value at left end-point
  };
  
}

#endif
