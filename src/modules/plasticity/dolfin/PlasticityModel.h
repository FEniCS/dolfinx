// Copyright (C) 2006 Kristian Oelgaard and Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-13

#ifndef __PLASTICITY_MODEL_H
#define __PLASTICITY_MODEL_H

#include <dolfin/uBlasDenseMatrix.h>
#include <dolfin/uBlasVector.h>
#include <dolfin/ublas.h>

namespace dolfin
{

  class PlasticityModel
  {
  public:

    /// Constructor
    PlasticityModel(real E, real nu);
    
    /// Destructor
    virtual ~PlasticityModel();

    /// Hardening parameter
    virtual real hardening_parameter(real const equivalent_plastic_strain) const;

    /// Equivalent plastic strain
    virtual real kappa(real equivalent_plastic_strain, const uBlasVector& current_stress, const real lambda_dot);

    /// Value of yield function f
    virtual real f(const uBlasVector& current_stress, const real equivalent_plastic_strain) = 0;

    /// First derivative of f with respect to sigma
    virtual void df(uBlasVector& df_dsigma, const uBlasVector& current_stress) = 0;

    /// First derivative of g with respect to sigma
    virtual void dg(uBlasVector& dg_dsigma, const uBlasVector& current_stress);
    
    /// Second derivative of g with respect to sigma
    virtual void ddg(uBlasDenseMatrix& ddg_ddsigma, const uBlasVector& current_stress) = 0;
    
    friend class ReturnMapping;
    friend class PlasticityProblem;

  private:

    /// Returns elastic tangent from the Lame coefficients
    uBlasDenseMatrix C_m(real lam, real mu);

    /// Model parameters
    real _hardening_parameter;
    uBlasDenseMatrix elastic_tangent;
  };
}

#endif
