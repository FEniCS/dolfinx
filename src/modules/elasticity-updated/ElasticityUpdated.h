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


      // Material parameters
      //real b = 0.1;
      //real E = 500.0;
      //real nu = 0.3;

      real b = 10.0;
      real b_p = 300.0;
      real E = 100.0;
      real nu = 0.3;


      real lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
      real mu = E / (2 * (1 + nu));
      
      // Compute sigma
      
      static Matrix
	gradwp(3, 3, Matrix::dense),
	gradwptransp(3, 3, Matrix::dense),
	sigma(3, 3, Matrix::dense),
	epsilon(3, 3, Matrix::dense),
	vsigma(3, 3, Matrix::dense),
	psigma(3, 3, Matrix::dense),
	F(3, 3, Matrix::dense),
	Ftransp(3, 3, Matrix::dense),
	F0transp(3, 3, Matrix::dense),
	tmp1(3, 3, Matrix::dense),
	tmp2(3, 3, Matrix::dense); 

      Matrix &sigma0 = *(sigma0array[cell_->id()]);
      Matrix &sigma1 = *(sigma1array[cell_->id()]);
      Matrix &vsigma1 = *(vsigmaarray[cell_->id()]);


      if(!computedsigma[cell_->id()])
      {
	gradwp(0, 0) = (wp(0).ddx())(mp);
	gradwp(0, 1) = (wp(0).ddy())(mp);
	gradwp(0, 2) = (wp(0).ddz())(mp);
	
	gradwp(1, 0) = (wp(1).ddx())(mp);
	gradwp(1, 1) = (wp(1).ddy())(mp);
	gradwp(1, 2) = (wp(1).ddz())(mp);
	
	gradwp(2, 0) = (wp(2).ddx())(mp);
	gradwp(2, 1) = (wp(2).ddy())(mp);
	gradwp(2, 2) = (wp(2).ddz())(mp);
	
	gradwp.transp(gradwptransp);
	
	Matrix &F0 = *(F0array[cell_->id()]);
	Matrix &F1 = *(F1array[cell_->id()]);
	
	//gradwp.mult(F0, F);
	F0.mult(gradwp, F);
	
	F *= k;
	F += F0;
	F1 = F; 


	//cout << "F1 on cell " << cell_->id() << ": " << endl;
	//F1.show();

	F.transp(Ftransp);
	F0.transp(F0transp);

	// Alternative 1

	///*
	epsilon = gradwp;
	epsilon += gradwptransp;
	//*/

	/*
	// Alternative 2

	Ftransp.mult(F, epsilon);
	epsilon(0, 0) -= 1;
	epsilon(1, 1) -= 1;
	epsilon(2, 2) -= 1;
	*/


	// Alternative 3

	/*
	Ftransp.mult(gradwptransp, tmp1);
	tmp1.mult(F, tmp2);

	epsilon = tmp2;
	
	Ftransp.mult(gradwp, tmp1);
	tmp1.mult(F, tmp2);

	epsilon += tmp2;
	*/


	//cout << "epsilon on cell " << cell_->id() << ": " << endl;
	//epsilon.show();
      
	
	//vsigma = epsilon;

	// Viscosity

	vsigma = gradwp;
	vsigma += gradwptransp;

	vsigma *= b;

	// Plasticity

	///*
	real psigmanorm;

	psigma = sigma0;
	psigmanorm = psigma.norm();
	//psigmanorm = 0.5;

	cout << "psigmanorm on cell " << cell_->id() << ": " << endl;
	cout << psigmanorm << endl;


	if(psigmanorm > 1)
	{
	  psigma *= -1.0 / psigmanorm;
	  psigma += sigma0;

	  psigma *= -1.0 / b_p;

	  epsilon += psigma;
	}
	//*/

	// Elasticity

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




	//cout << "sigma on cell " << cell_->id() << ": " << endl;
	//sigma.show();
      
      
	// Update sigma
	sigma *= k;
	sigma += sigma0;

	sigma1 = sigma; 
	vsigma1 = vsigma;


	//cout << "sigma0 on cell " << cell_->id() << ": " << endl;
	//sigma0->show();
	//cout << "sigma1 on cell " << cell_->id() << ": " << endl;
	//sigma1->show();

	computedsigma[cell_->id()] = true;
      }
      else
      {
	sigma = *(sigma1array[cell_->id()]);
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
    
  NewArray<Matrix *> sigma0array, sigma1array, vsigmaarray,
    F0array, F1array;
  NewArray<bool>     computedsigma;
  
 private:    
  ElementFunction::Vector f; // Source term
  ElementFunction::Vector wp;        // Velocity value at left end-point
  
};
  
}

#endif




