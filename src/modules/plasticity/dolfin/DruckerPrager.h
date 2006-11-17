// Copyright (C) 2006 Kristian Oelgaard and Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-13

#ifndef __Drucker_Prager_H
#define __Drucker_Prager_H

#include <dolfin/uBlasDenseMatrix.h>
#include <dolfin/uBlasVector.h>
#include <dolfin/PlasticityModel.h>

namespace dolfin
{

  class DruckerPrager : public PlasticityModel
  {

  public:

    /// Constructor
    DruckerPrager(real E, real nu, real friction_angle, real dilatancy_angle, 
                  real cohesion, real hardening_parameter);

    /// Hardening parameter
    real hardening_parameter(real const equivalent_plastic_strain) const;

    /// Value of yield function f
    real f(const uBlasVector& current_stress, const real equivalent_plastic_strain);

    /// First derivative of f with respect to sigma
    void df(uBlasVector& df_dsigma, const uBlasVector& current_stress);

    /// First derivative of g with respect to sigma
    void dg(uBlasVector& dg_dsigma, const uBlasVector& current_stress);

    /// Second derivative of g with respect to sigma
    void ddg(uBlasDenseMatrix& ddg_ddsigma, const uBlasVector& current_stress);

  private:

    /// Computes effective stresses
    real effective_stress(const uBlasVector& current_stress);

    /// First derivative of g with respect to sigma, excluding (+ alpha_dilatancy/3.0) on diagonal terms
    void dg_mod(uBlasVector& dg_dsigma_mod, const uBlasVector& current_stress);

    /// Model parameters
    real _effective_stress, _hardening_parameter;
    real alpha_friction, k_friction, alpha_dilatancy;

    /// Auxiliary variables
    uBlasDenseMatrix Am;
    uBlasVector dg_dsigma_mod;
  };
}

#endif
