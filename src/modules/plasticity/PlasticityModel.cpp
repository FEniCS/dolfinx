// Copyright (C) 2006 Kristian Oelgaard and Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-13

#include "dolfin/PlasticityModel.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
PlasticityModel::PlasticityModel() : _hardening_parameter(0.0)
{
  initialise();
}
//-----------------------------------------------------------------------------
PlasticityModel::~PlasticityModel()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void PlasticityModel::initialise()
{
  R.resize(6, 6, false);
  ddg_ddsigma.resize(6, 6, false);
  inverse_Q.resize(6, 6, false);
  I.resize(6);

  df_dsigma.resize(6, false);
  dg_dsigma.resize(6, false);
  sigma_current.resize(6, false);
  sigma_dot.resize(6, false);
  sigma_residual.resize(6, false);
  Rm.resize(6, false);
  Rn.resize(6, false);
  RinvQ.resize(6, false);

  residual_f = 0.0;
  delta_lambda = 0.0;
  lambda_dot = 0.0;
}
//-----------------------------------------------------------------------------
real PlasticityModel::hardening_parameter(real& eps_eq)
{
  return _hardening_parameter;
}
//-----------------------------------------------------------------------------
void PlasticityModel::dg(uBlasVector &dg_dsigma, uBlasVector& sigma)
{
  df(dg_dsigma, sigma);
}
//-----------------------------------------------------------------------------
void PlasticityModel::kappa(real& equivalent_plastic_strain, 
                uBlasVector &sigma, real& lambda_dot)
{
  equivalent_plastic_strain += lambda_dot;
}
//-----------------------------------------------------------------------------
void PlasticityModel::return_mapping(uBlasDenseMatrix& consistent_tangent, 
                                     uBlasDenseMatrix& elastic_tangent, 
                                     uBlasVector& sigma_trial, 
                                     uBlasVector& plastic_strain, 
                                     real& equivalent_plastic_strain)
{
  // Initialise variables
  delta_lambda = 0.0;
  sigma_residual.clear();
  sigma_current.assign(sigma_trial);

  // Compute hardening parameter
  _hardening_parameter = hardening_parameter(equivalent_plastic_strain);

  // Evaluate yield function (trial stresses)
  f(residual_f, sigma_current, equivalent_plastic_strain);

  // Check for yielding
  if (residual_f/norm_2(sigma_current) > 1.0e-12)
  {
    // compute normal vectors to yield surface and plastic potential
    df(df_dsigma, sigma_current);
    dg(dg_dsigma, sigma_current);

    uint iterations(0);

    // Newton iterations to project stress onto yield surface
    while ( std::abs(residual_f)/norm_2(sigma_current) > 1.0e-12)
    {
      iterations++;

      if (iterations > 50)
        dolfin_error("Plasticity return mapping iterations > 50.");

      // Compute second derivative (with respect to stresses) of plastic potential
      ddg(ddg_ddsigma, sigma_current);
  
      // Compute auxiliary matrix Q 
      inverse_Q.assign(I + delta_lambda*prod(elastic_tangent,ddg_ddsigma));

      // Invert Q
      inverse_Q.invert();

      // Compute auxiliary matrix R
      R.assign(prod(inverse_Q, elastic_tangent));

      // lambda_dot, increase of plastic multiplier
      lambda_dot = ( residual_f - inner_prod(sigma_residual, prod(inverse_Q, df_dsigma) ) ) / 
                        (inner_prod( df_dsigma, prod( R, dg_dsigma) ) + _hardening_parameter);

      // Compute stress increment            
      sigma_dot.assign(prod( (-lambda_dot*prod(elastic_tangent, dg_dsigma) -sigma_residual), inverse_Q ));

      // Increment plastic multiplier
      delta_lambda += lambda_dot;

      // Update current stress state
      sigma_current += sigma_dot;

      // Update equivalent plastic strain
      kappa(equivalent_plastic_strain, sigma_current, lambda_dot);

      // Compute hardening parameter
      _hardening_parameter = hardening_parameter(equivalent_plastic_strain);

      // Evaluate yield function at new stress state
      f(residual_f, sigma_current, equivalent_plastic_strain);

      // Normal to yield surface at new stress state
      df(df_dsigma, sigma_current);

      // Normal to plastic potential at new stress state
      dg(dg_dsigma, sigma_current);

      // Compute residual vector
      sigma_residual.assign(sigma_current - (sigma_trial - delta_lambda*prod(elastic_tangent,dg_dsigma) ) );
    } 

    // Update matrices
    ddg(ddg_ddsigma, sigma_current);

    // Compute auxiliary matrix Q
    inverse_Q.assign(I + delta_lambda*prod(elastic_tangent,ddg_ddsigma));

    // Invert Q
    inverse_Q.invert();

    // Compute auxiliary matrix R and vector Rn
    R.assign(prod(inverse_Q, elastic_tangent));
    Rn.assign(prod(R,df_dsigma));
      
    // Compute consistent tangent operator
    consistent_tangent.assign(R - prod( R, outer_prod(dg_dsigma, Rn) ) / 
              (inner_prod(df_dsigma, prod(R, dg_dsigma) ) + _hardening_parameter) );

    // Stresses for next Newton iteration, trial stresses are overwritten by current stresses
    sigma_trial.assign(sigma_current);

    // Update plastic strains
    plastic_strain += delta_lambda * dg_dsigma;

  } // end if plastic
}
//-----------------------------------------------------------------------------
