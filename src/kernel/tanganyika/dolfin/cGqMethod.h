// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __CGQ_METHOD_H
#define __CGQ_METHOD_H

#include <dolfin/Method.h>

namespace dolfin {

  class cGqMethod : public Method
  {
  public:
    
    cGqMethod(int iiHighestOrder) : Method(iiHighestOrder){
      // Initialize linearizations
      mLinearizations = new Matrix<real>* [iiHighestOrder+1];
      mSteps          = new Matrix<real>* [iiHighestOrder+1];
      mLinearizations[0] = new Matrix<real>(1,1);
      mSteps[0]          = new Matrix<real>(1,1);
      for (int i=1;i<(iiHighestOrder+1);i++){
	mLinearizations[i] = new Matrix<real>(i,i);
	mSteps[i]          = new Matrix<real>(i,1);	 
      }
    };	 
    
    ~cGqMethod(){
      // Clear linearizations
      for (int i=0;i<(iHighestOrder+1);i++){
	delete mLinearizations[i];
	delete mSteps[i];
      }
      delete mLinearizations;
      delete mSteps;
    }
        
    void Display();
    real GetWeightGeneric    (int iOrder, int iIndex, real dTau);
    real GetQuadratureWeight (int iOrder, int iNodalIndex);
    void UpdateLinearization (int iOrder, real dTimeStep, real dDerivative);
    void GetStep             (int iOrder, real *dStep);
    
  protected:
    
    void InitQuadrature                ();
    void GetNodalPoints                ();
    void ComputeNodalWeights           ();
    void ComputeInterpolationConstants ();
    void ComputeResidualFactors        ();
    void ComputeProductFactors         ();
    void ComputeQuadratureFactors      ();
    void ComputeInterpolationWeights   ();
    
  };

  typedef cGq cGqMethod;

}

#endif
