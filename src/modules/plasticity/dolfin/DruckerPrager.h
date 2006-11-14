// Copyright (C) 2006 Kristian Oelgaard and Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-13

#ifndef __Drucker_Prager_H
#define __Drucker_Prager_H

#include "dolfin/uBlasDenseMatrix.h"
#include "dolfin/uBlasVector.h"

#include "PlasticityModel.h"

namespace dolfin
{

  class DruckerPrager : public PlasticityModel
  {

  public:

    DruckerPrager(real& _phi, real& _psi, real& _c, real _H);

    void f(real& f0, uBlasVector& sig, real& eps_eq);

    real effective_stress(uBlasVector& sig);

    real hardening_parameter(real& eps_eq);

    real effective_stress();

    void df(uBlasVector& a, uBlasVector& sig);

    void dg(uBlasVector& b, uBlasVector& sig);

    void dg_mod(uBlasVector& b_mod, uBlasVector& sig, real sig_e);

    uBlasDenseMatrix A_m();

    void ddg(uBlasDenseMatrix& ddg_ddsigma, uBlasVector& sig);

  private:

      uBlasDenseMatrix Am;
      uBlasVector dg_dsigma_mod;
      real phi, psi, c, sig_e, alpha_f, k_f, alpha_g, H;
  };
}

#endif
