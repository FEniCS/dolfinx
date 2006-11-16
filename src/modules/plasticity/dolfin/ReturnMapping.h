// Copyright (C) 2006 Kristian Oelgaard and Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-13

#ifndef __RETURN_MAPPING_H
#define __RETURN_MAPPING_H

#include <dolfin/uBlasDenseMatrix.h>
#include <dolfin/uBlasVector.h>
#include <dolfin/ublas.h>
#include <dolfin/PlasticityModel.h>

namespace dolfin
{

  class ReturnMapping
  {
  public:

    /// Constructor
    ReturnMapping();

    /// Initialise variables
    void initialise();    

    /// Closest point projection return mapping
    void ClosestPoint(PlasticityModel& plastic_model, uBlasDenseMatrix& consistent_tangent,
                      uBlasVector& trial_stresses, uBlasVector& plastic_strain, real& equivalent_plastic_strain);

  private:

    /// Variables for return mapping
    uBlasVector df_dsigma, dg_dsigma, sigma_current, sigma_dot, sigma_residual;
    real residual_f, delta_lambda, lambda_dot, hardening_parameter;

    /// Auxiliary variables to speed up return mapping
    uBlasDenseMatrix R;
    uBlasDenseMatrix ddg_ddsigma;
    uBlasDenseMatrix inverse_Q;
    ublas::identity_matrix<real> I;
    uBlasVector Rm, Rn, RinvQ;
  };
}

#endif
