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

    VonMises(real& yield_stress, real& hardening_parameter);

    real hardening_parameter(real eps_eq);

    real f(uBlasVector& sig, real eps_eq);

    void df(uBlasVector& a, uBlasVector& sig);

    void ddg(uBlasDenseMatrix& ddg_ddsigma, uBlasVector& sig);

    real effective_stress(uBlasVector& sig);

    uBlasDenseMatrix A_m();

  private:

    uBlasDenseMatrix Am;
    uBlasVector dg_dsigma;

    real sig_o, sig_e, H;
  };
}
#endif
