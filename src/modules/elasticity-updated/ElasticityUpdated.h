// Copyright (C) 2003 Fredrik Bengzon and Johan Jansson.
// Licensed under the GNU GPL Version 2.

#ifndef __ELASTICITY_UPDATED2_H
#define __ELASTICITY_UPDATED2_H

#include <dolfin/PDE.h>


namespace dolfin {
  
class ElasticityUpdated : public PDE {
public:
    
  ElasticityUpdated(Function::Vector& source,
		     Function::Vector& wprevious) : PDE(3, 3), f(3), wp(3)
    {
      add(f,  source);
      add(wp, wprevious);
    }
    
  real lhs(ShapeFunction::Vector& u, ShapeFunction::Vector& v)
    {
      return
	(u(0) * v(0) + u(1) * v(1) + u(2) * v(2)) * dx;
    }
    
  real rhs(ShapeFunction::Vector& v)
    {
      const Point &mp = cell_->midpoint();

      // Debug

      /*
      // Derivatives
      ElementFunction wx0 = wp(0).ddx();
      ElementFunction wy0 = wp(0).ddy();
      ElementFunction wz0 = wp(0).ddz();

      ElementFunction wx1 = wp(1).ddx();
      ElementFunction wy1 = wp(1).ddy();
      ElementFunction wz1 = wp(1).ddz();

      ElementFunction wx2 = wp(2).ddx();
      ElementFunction wy2 = wp(2).ddy();
      ElementFunction wz2 = wp(2).ddz();

      std::cout << "wp(0).ddx(): " << wx0(mp) << std::endl;
      std::cout << "wp(0).ddy(): " << wy0(cell_->coord(0)) << std::endl;
      std::cout << "wp(0).ddz(): " << wz0(cell_->coord(0)) << std::endl;

      std::cout << "wp(1).ddx(): " << wx1(cell_->coord(0)) << std::endl;
      std::cout << "wp(1).ddy(): " << wy1(cell_->coord(0)) << std::endl;
      std::cout << "wp(1).ddz(): " << wz1(cell_->coord(0)) << std::endl;

      std::cout << "wp(2).ddx(): " << wx2(cell_->coord(0)) << std::endl;
      std::cout << "wp(2).ddy(): " << wy2(cell_->coord(0)) << std::endl;
      std::cout << "wp(2).ddz(): " << wz2(cell_->coord(0)) << std::endl;
      */

      // Material parameters
      
      real b = 0.05;

      real E = 500.0;
      real nu = 0.3;
      
      real lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
      real mu = E / (2 * (1 + nu));
      
      //real lambda = 20.0;
      //real mu = 5.0;
      
      // Compute sigma
      
      static Matrix sigma(3, 3), epsilon(3, 3), vsigma(3, 3); 

      if(!computedsigma[cell_->id()])
      {
	
	static ElementFunction
	  epsilon00, epsilon01, epsilon02,
	  epsilon10, epsilon11, epsilon12,
	  epsilon20, epsilon21, epsilon22;
	
	epsilon(0, 0) = (wp(0).ddx() + wp(0).ddx())(mp);
	epsilon(0, 1) = (wp(0).ddy() + wp(1).ddx())(mp);
	epsilon(0, 2) = (wp(0).ddz() + wp(2).ddx())(mp);
	
	epsilon(1, 0) = (wp(1).ddx() + wp(0).ddy())(mp);
	epsilon(1, 1) = (wp(1).ddy() + wp(1).ddy())(mp);
	epsilon(1, 2) = (wp(1).ddz() + wp(2).ddy())(mp);
	
	epsilon(2, 0) = (wp(2).ddx() + wp(0).ddz())(mp);
	epsilon(2, 1) = (wp(2).ddy() + wp(1).ddz())(mp);
	epsilon(2, 2) = (wp(2).ddz() + wp(2).ddz())(mp);
	
	
	//cout << "epsilon on cell " << cell_->id() << ": " << endl;
	//epsilon.show();
      
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
	
	vsigma = epsilon;
	vsigma *= b;

	//cout << "sigma on cell " << cell_->id() << ": " << endl;
	//sigma.show();
      
	Matrix *sigma0, *sigma1; // *vsigma1;
	
	sigma0 = sigma0array[cell_->id()];
	sigma1 = sigma1array[cell_->id()];
      
	sigma *= k;
	sigma += *sigma0;
	*sigma1 = sigma; 

	*(vsigmaarray[cell_->id()]) = vsigma;
	
	//cout << "sigma0 on cell " << cell_->id() << ": " << endl;
	//sigma0->show();
	//cout << "sigma1 on cell " << cell_->id() << ": " << endl;
	//sigma1->show();

	computedsigma[cell_->id()] = true;
      }
      else
      {
	Matrix *sigma1;
	sigma1 = sigma1array[cell_->id()];

	sigma = *sigma1;
	vsigma = *(vsigmaarray[cell_->id()]);
      }

	
      return ((wp(0) * v(0) + wp(1) * v(1) + wp(2) * v(2)) +
	      k * (f(0) * v(0) + f(1) * v(1) + f(2) * v(2)) -
	      k * (sigma(0, 0) * v(0).ddx() +
		   sigma(0, 1) * v(0).ddy() +
		   sigma(0, 2) * v(0).ddz() +
		   sigma(1, 0) * v(1).ddx() +
		   sigma(1, 1) * v(1).ddy() +
		   sigma(1, 2) * v(1).ddz() +
		   sigma(2, 0) * v(2).ddx() +
		   sigma(2, 1) * v(2).ddy() +
		   sigma(2, 2) * v(2).ddz()) -
	      k * (vsigma(0, 0) * v(0).ddx() +
		   vsigma(0, 1) * v(0).ddy() +
		   vsigma(0, 2) * v(0).ddz() +
		   vsigma(1, 0) * v(1).ddx() +
		   vsigma(1, 1) * v(1).ddy() +
		   vsigma(1, 2) * v(1).ddz() +
		   vsigma(2, 0) * v(2).ddx() +
		   vsigma(2, 1) * v(2).ddy() +
		   vsigma(2, 2) * v(2).ddz())) * dx;

      //return (wp(0) * v(0) + wp(1) * v(1) + wp(2) * v(2)) * dx;
      }
    
  NewArray<Matrix *> sigma0array, sigma1array, vsigmaarray;
  NewArray<bool>     computedsigma;
  
 private:    
  ElementFunction::Vector f; // Source term
  ElementFunction::Vector wp;        // Velocity value at left end-point
  
};
  
}

#endif




