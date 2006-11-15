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
    PlasticityModel();
    
    /// Destructor
    virtual ~PlasticityModel();

    /// Initialise variables
    void initialise();    

    /// Hardening parameter
    virtual real hardening_parameter(real eps_eq);

    /// Equivalent plastic strain
    virtual real kappa(real eps_eq, uBlasVector& sig, real lambda);

    /// Value of yield function f
    virtual real f(uBlasVector& sig, real eps_eq) = 0;

    /// First derivative of f with respect to sigma
    virtual void df(uBlasVector& a, uBlasVector& sig) = 0;

    /// First derivative of g with respect to sigma
    virtual void dg(uBlasVector& b, uBlasVector& sig);
    
    /// Second derivative of g with respect to sigma
    virtual void ddg(uBlasDenseMatrix& dm_dsig, uBlasVector& sig) = 0;
    
    friend class ReturnMapping;

  private:

    uBlasDenseMatrix R;
    uBlasDenseMatrix ddg_ddsigma;

    uBlasDenseMatrix inverse_Q;
    ublas::identity_matrix<real> I;

    uBlasVector df_dsigma, dg_dsigma, sigma_current, sigma_dot, sigma_residual, Rm, Rn, RinvQ;

    real _hardening_parameter;

  };
}

#endif
