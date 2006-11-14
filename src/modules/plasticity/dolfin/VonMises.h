// Copyright (C) 2006 Kristian Oelgaard and Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-13

#ifndef __VON_MISES_H
#define __VON_MISES_H

#include "dolfin/uBlasDenseMatrix.h"
#include "dolfin/uBlasVector.h"

#include "PlasticityModel.h"


namespace dolfin
{

  class VonMises : public PlasticityModel
  {
  public:

    VonMises(real& yield_stress, real& hardening_parameter);

    void effective_stress(real& sig_e, uBlasVector& sig);

    real effective_stress();

    void df(uBlasVector& a, uBlasVector& sig);

    uBlasDenseMatrix A_m();

    void ddg(uBlasDenseMatrix& ddg_ddsigma, uBlasVector& sig);

    void f(real& f0, uBlasVector& sig, real& eps_eq);

    real hardening_parameter(real& eps_eq);

  private:

    uBlasDenseMatrix Am;
    uBlasVector dg_dsigma;

    real sig_o, sig_e, H;
  };
}
#endif
