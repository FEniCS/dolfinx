// Copyright (C) 2006 Kristian Oelgaard and Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-13

#ifndef __PLASTICITY_MODEL_H
#define __PLASTICITY_MODEL_H

#include <dolfin/uBlasDenseMatrix.h>
#include <dolfin/uBlasVector.h>

namespace dolfin
{

  class PlasticityModel
  {
  public:

    /// Constructor
    PlasticityModel(const real E, const real nu);
    
    /// Destructor
    virtual ~PlasticityModel();

    /// Hardening parameter
    virtual real hardening_parameter(real const equivalent_plastic_strain) const;

    /// Equivalent plastic strain
    virtual real kappa(real equivalent_plastic_strain, 
                       const uBlasVector& stress, const real lambda_dot) const;

    /// Value of yield function f
    virtual real f(const uBlasVector& stress, const real equivalent_plastic_strain) = 0;

    /// First derivative of f with respect to sigma
    virtual void df(uBlasVector& df_dsigma, const uBlasVector& stress) = 0;

    /// First derivative of g with respect to sigma
    virtual void dg(uBlasVector& dg_dsigma, const uBlasVector& stress);
    
    /// Second derivative of g with respect to sigma
    virtual void ddg(uBlasDenseMatrix& ddg_ddsigma, const uBlasVector& stress) = 0;
    
    friend class ReturnMapping;
    friend class PlasticityProblem;

  private:

    /// Model parameters
    real _hardening_parameter;
    uBlasDenseMatrix elastic_tangent;
  };
}

#endif
