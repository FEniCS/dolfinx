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
    virtual real hardening_parameter(real& eps_eq);

    /// Value of yield function f
    virtual void f(real& f0, uBlasVector& sig, real& eps_eq) = 0;

    /// First derivative of f with respect to sigma
    virtual void df(uBlasVector& a, uBlasVector& sig) = 0;

    /// First derivative of g with respect to sigma
    virtual void dg(uBlasVector& b, uBlasVector& sig);
    
    /// Second derivative of g with respect to sigma
    virtual void ddg(uBlasDenseMatrix& dm_dsig, uBlasVector& sig) = 0;

    /// Equivalent plastic strain
    virtual void kappa(real& eps_eq, uBlasVector& sig, real& lambda);
    
    /// Closest point projection return mapping
    void return_mapping(uBlasDenseMatrix& cons_t, uBlasDenseMatrix& D, 
        uBlasVector& t_sig, uBlasVector& eps_p, real& eps_eq);

  private:

    uBlasDenseMatrix R;
    uBlasDenseMatrix ddg_ddsigma;

    uBlasDenseMatrix inverse_Q;
    ublas::identity_matrix<real> I;

    uBlasVector df_dsigma, dg_dsigma, sigma_current, sigma_dot, sigma_residual, Rm, Rn, RinvQ;

    real residual_f, _hardening_parameter, delta_lambda, lambda_dot;
  
  };
}

#endif
