// Copyright (C) 2006 Kristian Oelgaard and Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-13

#ifndef __VON_MISES_H
#define __VON_MISES_H

#include <dolfin/uBlasDenseMatrix.h>
#include <dolfin/uBlasVector.h>
#include <dolfin/PlasticityModel.h>


namespace dolfin
{

  class VonMises : public PlasticityModel
  {
  public:

    /// Constructor
    VonMises(real E, real nu, real yield_stress, real hardening_parameter);

    /// Hardening parameter
    real hardening_parameter(const real equivalent_plastic_strain) const;

    /// Value of yield function f
    real f(const uBlasVector& current_stress, const real equivalent_plastic_strain);

    /// First derivative of f with respect to sigma
    void df(uBlasVector& df_dsigma, const uBlasVector& current_stress);

    /// Second derivative of g with respect to sigma
    void ddg(uBlasDenseMatrix& ddg_ddsigma, const uBlasVector& current_stress);

  private:

    /// Computes effective stresses
    real effective_stress(const uBlasVector& current_stress);

    /// Returns auxiliary matrix Am
    uBlasDenseMatrix A_m();

    /// Model parameters
    real _yield_stress, _effective_stress, _hardening_parameter;

    /// Auxiliary variables
    uBlasDenseMatrix Am;
    uBlasVector dg_dsigma;
  };
}
#endif
