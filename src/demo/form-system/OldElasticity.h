// Copyright (C) 2003 Fredrik Bengzon and Johan Jansson.
// Licensed under the GNU GPL Version 2.

#ifndef __OLDELASTICITY_H
#define __OLDELASTICITY_H

#include <dolfin/PDE.h>

namespace dolfin {
  
  class OldElasticity : public PDE {
  public:
    
    OldElasticity() :
      PDE(2, 3)
      {
      }
    
    real lhs(ShapeFunction::Vector& u, ShapeFunction::Vector& v)
    {
      //real b = 0.1;
      //real E = 100.0;
      //real nu = 0.3;

      //real lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
      //real mu = E / (2 * (1 + nu));

      real lambda = 1.0;
      real mu = 1.0;

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

//       return
//  	(sigma(0, 0) * epsilonv(0, 0) +
//  	 sigma(0, 1) * epsilonv(0, 1) +
//  	 sigma(0, 2) * epsilonv(0, 2) +
//  	 sigma(1, 0) * epsilonv(1, 0) +
//  	 sigma(1, 1) * epsilonv(1, 1) +
//  	 sigma(1, 2) * epsilonv(1, 2) +
//  	 sigma(2, 0) * epsilonv(2, 0) +
//  	 sigma(2, 1) * epsilonv(2, 1) +
//  	 sigma(2, 2) * epsilonv(2, 2)) *
//  	dx;

       return
  	(epsilon(0, 0) * epsilonv(0, 0) +
  	 epsilon(0, 1) * epsilonv(0, 1) +
 	 epsilon(0, 2) * epsilonv(0, 2) +
 	 epsilon(1, 0) * epsilonv(1, 0) +
 	 epsilon(1, 1) * epsilonv(1, 1) +
 	 epsilon(1, 2) * epsilonv(1, 2) +
 	 epsilon(2, 0) * epsilonv(2, 0) +
 	 epsilon(2, 1) * epsilonv(2, 1) +
 	 epsilon(2, 2) * epsilonv(2, 2)) *
 	dx;


       //*/
    }
    
    /*
    real rhs(ShapeFunction::Vector& v)
    {
      return (f(0) * v(0) + f(1) * v(1) + f(2) * v(2)) * dx;
    }
    */
  };
  
}

#endif
