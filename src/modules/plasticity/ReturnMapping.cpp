// Copyright (C) 2006 Kristian Oelgaard and Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-13

#include <dolfin/ReturnMapping.h>

using namespace dolfin;

void ReturnMapping::ClosestPoint(PlasticityModel* plas, uBlasDenseMatrix& consistent_tangent, 
                                     uBlasDenseMatrix& elastic_tangent, 
                                     uBlasVector& sigma_trial, 
                                     uBlasVector& plastic_strain, 
                                     real& equivalent_plastic_strain)
{
  real residual_f(0.0), delta_lambda(0.0), lambda_dot(0.0);
  // Initialise variables
//  delta_lambda = 0.0;
  plas->sigma_residual.clear();
  plas->sigma_current.assign(sigma_trial);

  // Compute hardening parameter
  plas->_hardening_parameter = plas->hardening_parameter(equivalent_plastic_strain);

  // Evaluate yield function (trial stresses)
  residual_f = plas->f(plas->sigma_current, equivalent_plastic_strain);

  // Check for yielding
  if (residual_f/norm_2(plas->sigma_current) > 1.0e-12)
  {
    // compute normal vectors to yield surface and plastic potential
    plas->df(plas->df_dsigma, plas->sigma_current);
    plas->dg(plas->dg_dsigma, plas->sigma_current);

    uint iterations(0);

    // Newton iterations to project stress onto yield surface
    while ( std::abs(residual_f)/norm_2(plas->sigma_current) > 1.0e-12)
    {
      iterations++;

      if (iterations > 50)
        dolfin_error("Return mapping iterations > 50.");

      // Compute second derivative (with respect to stresses) of plastic potential
      plas->ddg(plas->ddg_ddsigma, plas->sigma_current);
  
      // Compute auxiliary matrix Q 
      plas->inverse_Q.assign(plas->I + delta_lambda*prod(elastic_tangent,plas->ddg_ddsigma));

      // Invert Q
      plas->inverse_Q.invert();

      // Compute auxiliary matrix R
      plas->R.assign(prod(plas->inverse_Q, elastic_tangent));

      // lambda_dot, increase of plastic multiplier
      lambda_dot = (residual_f - inner_prod(plas->sigma_residual, prod(plas->inverse_Q, plas->df_dsigma) ) ) / 
                        (inner_prod( plas->df_dsigma, prod( plas->R, plas->dg_dsigma) ) + plas->_hardening_parameter);

      // Compute stress increment            
      plas->sigma_dot.assign(prod( (-lambda_dot*prod(elastic_tangent, plas->dg_dsigma) -plas->sigma_residual), plas->inverse_Q ));

      // Increment plastic multiplier
      delta_lambda += lambda_dot;

      // Update current stress state
      plas->sigma_current += plas->sigma_dot;

      // Update equivalent plastic strain
      equivalent_plastic_strain = plas->kappa(equivalent_plastic_strain, plas->sigma_current, lambda_dot);

      // Compute hardening parameter
      plas->_hardening_parameter = plas->hardening_parameter(equivalent_plastic_strain);

      // Evaluate yield function at new stress state
      residual_f = plas->f(plas->sigma_current, equivalent_plastic_strain);

      // Normal to yield surface at new stress state
      plas->df(plas->df_dsigma, plas->sigma_current);

      // Normal to plastic potential at new stress state
      plas->dg(plas->dg_dsigma, plas->sigma_current);

      // Compute residual vector
      plas->sigma_residual.assign(plas->sigma_current - (sigma_trial - delta_lambda*prod(elastic_tangent,plas->dg_dsigma) ) );
    } 

    // Update matrices
    plas->ddg(plas->ddg_ddsigma, plas->sigma_current);

    // Compute auxiliary matrix Q
    plas->inverse_Q.assign(plas->I + delta_lambda*prod(elastic_tangent,plas->ddg_ddsigma));

    // Invert Q
    plas->inverse_Q.invert();

    // Compute auxiliary matrix R and vector Rn
    plas->R.assign(prod(plas->inverse_Q, elastic_tangent));
    plas->Rn.assign(prod(plas->R,plas->df_dsigma));
      
    // Compute consistent tangent operator
    consistent_tangent.assign(plas->R - prod( plas->R, outer_prod(plas->dg_dsigma, plas->Rn) ) / 
              (inner_prod(plas->df_dsigma, prod(plas->R, plas->dg_dsigma) ) + plas->_hardening_parameter) );

    // Stresses for next Newton iteration, trial stresses are overwritten by current stresses
    sigma_trial.assign(plas->sigma_current);

    // Update plastic strains
    plastic_strain += delta_lambda * plas->dg_dsigma;

  } // end if plastic
}
//-----------------------------------------------------------------------------
