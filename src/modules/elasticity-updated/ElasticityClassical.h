// Copyright (C) 2003 Fredrik Bengzon and Johan Jansson.
// Licensed under the GNU GPL Version 2.

#ifndef __ELASTICITY_H
#define __ELASTICITY_H

#include <dolfin/PDE.h>

namespace dolfin {
  
  class ElasticityClassical : public PDE {
  public:
    
    ElasticityClassical(Function::Vector& source) :
      PDE(3, 3), f(3)
    {
      add(f,  source);
    }
    
    real lhs(ShapeFunction::Vector& u, ShapeFunction::Vector& v)
    {
      //real b = 0.1;
      real E = 100.0;
      real nu = 0.3;

      real lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
      real mu = E / (2 * (1 + nu));

      /*

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
	 sigma20 * epsilonv20 + sigma21 * epsilonv21 + sigma22 * epsilonv22) *
	dx;
      */

      ///*
      const Point &mp = cell_->midpoint();

      static Matrix
        gradu(3, 3, Matrix::dense),
        gradutransp(3, 3, Matrix::dense),
        gradv(3, 3, Matrix::dense),
        gradvtransp(3, 3, Matrix::dense),
	epsilon(3, 3, Matrix::dense),
	epsilonv(3, 3, Matrix::dense),
	sigma(3, 3, Matrix::dense);

      gradu(0, 0) = (u(0).ddx())(mp);
      gradu(0, 1) = (u(0).ddy())(mp);
      gradu(0, 2) = (u(0).ddz())(mp);
      
      gradu(1, 0) = (u(1).ddx())(mp);
      gradu(1, 1) = (u(1).ddy())(mp);
      gradu(1, 2) = (u(1).ddz())(mp);
      
      gradu(2, 0) = (u(2).ddx())(mp);
      gradu(2, 1) = (u(2).ddy())(mp);
      gradu(2, 2) = (u(2).ddz())(mp);
      
      gradu.transp(gradutransp);

      gradv(0, 0) = (v(0).ddx())(mp);
      gradv(0, 1) = (v(0).ddy())(mp);
      gradv(0, 2) = (v(0).ddz())(mp);
      
      gradv(1, 0) = (v(1).ddx())(mp);
      gradv(1, 1) = (v(1).ddy())(mp);
      gradv(1, 2) = (v(1).ddz())(mp);
      
      gradv(2, 0) = (v(2).ddx())(mp);
      gradv(2, 1) = (v(2).ddy())(mp);
      gradv(2, 2) = (v(2).ddz())(mp);
      
      gradu.transp(gradutransp);

      epsilon = gradu;
      epsilon += gradutransp;

      epsilonv = gradv;
      epsilonv += gradvtransp;

      sigma(0, 0) =
	(lambda + 2 * mu) * epsilon(0, 0) +
	lambda * epsilon(1, 1) + lambda * epsilon(2, 2);
      sigma(1, 1) =
	(lambda + 2 * mu) * epsilon(1, 1) +
	lambda * epsilon(0, 0) + lambda * epsilon(2, 2);
      sigma(2, 2) =
	(lambda + 2 * mu) * epsilon(2, 2) +
	lambda * epsilon(0, 0) + lambda * epsilon(1, 1);
      sigma(1, 2) = mu * epsilon(1, 2);
      sigma(2, 1) = sigma(1, 2);
      sigma(2, 0) = mu * epsilon(2, 0);
      sigma(0, 2) = sigma(2, 0); 
      sigma(0, 1) = mu * epsilon(0, 1);
      sigma(1, 0) = sigma(0, 1);

      return
	(sigma(0, 0) * epsilonv(0, 0) +
	 sigma(0, 1) * epsilonv(0, 1) +
	 sigma(0, 2) * epsilonv(0, 2) +
	 sigma(1, 0) * epsilonv(1, 0) +
	 sigma(1, 1) * epsilonv(1, 1) +
	 sigma(1, 2) * epsilonv(1, 2) +
	 sigma(2, 0) * epsilonv(2, 0) +
	 sigma(2, 1) * epsilonv(2, 1) +
	 sigma(2, 2) * epsilonv(2, 2)) *
	dx;
      //*/
    }
    
    real rhs(ShapeFunction::Vector& v)
    {
      return (f(0) * v(0) + f(1) * v(1) + f(2) * v(2)) * dx;
    }
    
  private:    
    ElementFunction::Vector f; // Source term
  };
  
}

#endif
