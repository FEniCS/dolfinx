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

    DruckerPrager(real _phi, real _psi, real _c, real _H);

    real hardening_parameter(real eps_eq);

    real f(uBlasVector& sig, real eps_eq);

    void df(uBlasVector& a, uBlasVector& sig);

    void dg(uBlasVector& b, uBlasVector& sig);

    void ddg(uBlasDenseMatrix& ddg_ddsigma, uBlasVector& sig);

    real effective_stress(uBlasVector& sig);

    void dg_mod(uBlasVector& b_mod, uBlasVector& sig, real sig_e);

    uBlasDenseMatrix A_m();

  private:

      uBlasDenseMatrix Am;
      uBlasVector dg_dsigma_mod;
      real phi, psi, c, sig_e, alpha_f, k_f, alpha_g, H;
  };
}

#endif
