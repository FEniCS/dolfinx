// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __METHOD_H
#define __METHOD_H

#include <dolfin/constants.h>
#include <dolfin/precalc.h>

namespace dolfin {

  class Method {
  public:

    Method();
    virtual ~Method();
    
    void init(int order);
    
    real GetNodalPoint            (int iOrder, int iNodalIndex);
    real GetBasisFunction         (int iOrder, int iIndex, real dTau);
    real GetDerivativeRight       (int iOrder, int iIndex);
    real GetStabilityWeight       (int iOrder, int iIndex);
    real GetWeight                (int iOrder, int iIndex, int iNodalPoint);
    real GetInterpolationConstant (int iOrder);
    real GetResidualFactor        (int iOrder);
    real GetProductFactor         (int iOrder);
    real GetQuadratureFactor      (int iOrder);
    real GetInterpolationPoint    (int iOrder, int iNodalIndex);
    real GetInterpolationWeight   (int iOrder, int iNodalIndex);
    
    bool DataOK();
    
    virtual void Display() = 0;
    virtual real GetWeightGeneric    (int iOrder, int iIndex, real dTau) = 0;
    virtual real GetQuadratureWeight (int iOrder, int iNodalIndex)         = 0;
    virtual void UpdateLinearization (int iOrder, real dTimeStep, real dDerivative) = 0;
    virtual void GetStep             (int iOrder, real *dStep) = 0;
    
  protected:
    
    void   CheckWeight                 (real dWeight);
    void   ComputeNodalBasis           ();
    void   ComputeDerivativeWeights    ();
    real Value                       (real *dVals, int iOrder, real dTau);
    
    virtual void InitQuadrature                () = 0;
    virtual void GetNodalPoints                () = 0;
    virtual void ComputeNodalWeights           () = 0;
    virtual void ComputeInterpolationConstants () = 0;
    virtual void ComputeResidualFactors        () = 0;
    virtual void ComputeProductFactors         () = 0;
    virtual void ComputeQuadratureFactors      () = 0;
    virtual void ComputeInterpolationWeights   () = 0;
    
    Quadrature   *qQuadrature;     // Quadrature for basis functions
    Quadrature   *qGauss;          // Quadrature for interpolation
    Lagrange   ***lBasisFunctions;
    
    Matrix<real> **mLinearizations;
    Matrix<real> **mSteps;
    
    bool *bLinearizationSingular;
    
    int       iHighestOrder;
    real  **dNodalPoints; 
    real ***dNodalWeights;
    real ***dWeightFunctionNodalValues;
    real  **dDerivativeWeights;
    real  **dStabilityWeights;
    real   *dInterpolationConstants;
    real   *dResidualFactors;
    real   *dProductFactors;
    real   *dQuadratureFactors;
    real  **dInterpolationPoints;
    real  **dInterpolationWeights;
    
    bool ok;
  
  };

}

#endif
