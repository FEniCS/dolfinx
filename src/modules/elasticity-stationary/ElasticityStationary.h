// Copyright (C) 2003 Fredrik Bengzon and Johan Jansson.
// Licensed under the GNU GPL Version 2.

#ifndef __ELASTICITYSTATIONARY_H
#define __ELASTICITYSTATIONARY_H

#include <dolfin/PDE.h>

namespace dolfin {
  
  class ElasticityStationary : public PDE {
  public:
    
    ElasticityStationary(Function::Vector& source,
	       Function::Vector& uprevious,
	       Function::Vector& wprevious,
	       Function& diffusion,
	       Function::Vector& convection) : PDE(3, 3)
    {
      add(f,  source);
      add(up, uprevious);
      add(wp, wprevious);
      add(a,  diffusion);
      add(b,  convection);
    }
    
    real lhs(ShapeFunction::Vector& u, ShapeFunction::Vector& v)
    {
      real E = 1.0;
      real nu = 0.3;

      real lambda = nu * E / ((1 + nu) * (1 - 2 * nu));
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
	(sigma00 * epsilonv00 + sigma01 * epsilonv01 + sigma02 * epsilonv02 +
	 sigma10 * epsilonv10 + sigma11 * epsilonv11 + sigma12 * epsilonv12 +
	 sigma20 * epsilonv20 + sigma21 * epsilonv21 + sigma22 * epsilonv22)
	* dK;

    }
    
    real rhs(ShapeFunction::Vector& v)
    {
      return (f(0) * v(0) + f(1) * v(1) + f(2) * v(2)) * dK;
    }
    
 private:    
    ElementFunction::Vector f;  // Source term
    ElementFunction::Vector up; // Displacement value at left end-point
    ElementFunction::Vector wp; // Velocity value at left end-point
    ElementFunction a;          // Diffusivity
    ElementFunction::Vector b;  // Convection
  };
  
}

#endif
