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
    real f(const uBlasVector& stress, const real equivalent_plastic_strain);

    /// First derivative of f with respect to sigma
    void df(uBlasVector& df_dsigma, const uBlasVector& stress);

    /// Second derivative of g with respect to sigma
    void ddg(uBlasDenseMatrix& ddg_ddsigma, const uBlasVector& stress);

  private:

    /// Computes effective stresses
    real effective_stress(const uBlasVector& stress);

    /// Model parameters
    real _yield_stress, _effective_stress, _hardening_parameter;

    /// Auxiliary variables
    uBlasDenseMatrix A;
    uBlasVector dg_dsigma;
  };
}
#endif
