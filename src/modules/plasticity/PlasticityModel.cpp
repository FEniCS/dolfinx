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

// Initialise variables
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

// If no hardening law is defined
double PlasticityModel::hardening_parameter(double &eps_eq)
{
  return _hardening_parameter;
}

// If dg is not supplied by user, associative flow is assumed
void PlasticityModel::dg(ublas::vector <double> &dg_dsigma, ublas::vector <double> &sigma)
{
  df(dg_dsigma, sigma);
}

// Equivalent plastic strain is as default equal to the plastic multiplier
void PlasticityModel::kappa(double &equivalent_plastic_strain, ublas::vector<double> &sigma, double &lambda_dot)
{
  equivalent_plastic_strain += lambda_dot;
}

// Return mapping algorithm
void PlasticityModel::return_mapping(ublas::matrix <double> &consistent_tangent, ublas::matrix <double> &elastic_tangent, ublas::vector <double> &sigma_trial, ublas::vector <double> &plastic_strain, double &equivalent_plastic_strain)
{
  // initialising variables
  delta_lambda = 0.0;
  sigma_residual.clear();
  sigma_current.assign(sigma_trial);

  // computing hardening parameter
  _hardening_parameter = hardening_parameter(equivalent_plastic_strain);

  // value of yield function (trial stresses)
  f(residual_f, sigma_current, equivalent_plastic_strain);

  // check if yielding occurs
  if (residual_f/norm_2(sigma_current) > 1.0e-12)
  {
    // compute normal vectors to yield surface and plastic potential
    df(df_dsigma, sigma_current);
    dg(dg_dsigma, sigma_current);

    int iterations(0);

    // iterate
    while ( std::abs(residual_f)/norm_2(sigma_current) > 1.0e-12)
    {
      iterations++;

      if (iterations>10)
      {
        cout << "return mapping iterations > 10" << endl;
        break;
      }

      // compute second derivative (with respect to stresses) of plastic potential
      ddg(ddg_ddsigma, sigma_current);
  
      // computing auxiliary matrix Q 
      inverse_Q.assign(I + delta_lambda*prod(elastic_tangent,ddg_ddsigma));

      // inverting Q
      inverse_Q.invert();

      // compute auxiliary matrix R
      R.assign(prod(inverse_Q, elastic_tangent));

      // lambda_dot, increase of plastic multiplier
      lambda_dot = ( residual_f - inner_prod(sigma_residual, prod(inverse_Q, df_dsigma) ) ) / (inner_prod( df_dsigma, prod( R, dg_dsigma) ) + _hardening_parameter);

      // stress increment            
      sigma_dot.assign(prod( (-lambda_dot*prod(elastic_tangent, dg_dsigma) -sigma_residual), inverse_Q ));

      // incrementing plastic multiplier
      delta_lambda += lambda_dot;

      // update current stress state
      sigma_current += sigma_dot;

      // update equivalent plastic strain
      kappa(equivalent_plastic_strain, sigma_current, lambda_dot);

      // compute hardening parameter
      _hardening_parameter = hardening_parameter(equivalent_plastic_strain);

      // value of yield function at new stress state
      f(residual_f, sigma_current, equivalent_plastic_strain);

      // normal vector to yield surface at new stress state
      df(df_dsigma, sigma_current);

      // normal vector to plastic potential at new stress state
      dg(dg_dsigma, sigma_current);

      // compute residual vector
      sigma_residual.assign(sigma_current - (sigma_trial - delta_lambda*prod(elastic_tangent,dg_dsigma) ) );

    } // end while

    // updating matrices
    ddg(ddg_ddsigma, sigma_current);

    // computing auxiliary matrix Q
    inverse_Q.assign(I + delta_lambda*prod(elastic_tangent,ddg_ddsigma));

    // inverting Q
    inverse_Q.invert();

    // compute auxiliary matrix R and vector Rn
    R.assign(prod(inverse_Q, elastic_tangent));
    Rn.assign(prod(R,df_dsigma));
      
    // compute consistent tangent operator
    consistent_tangent.assign(R - prod( R, outer_prod(dg_dsigma, Rn) )/(inner_prod(df_dsigma, prod(R, dg_dsigma) ) + _hardening_parameter) );

    // stresses for next Newton iteration, trial stresses are overwritten by current stresses
    sigma_trial.assign(sigma_current);

    // update plastic strains
    plastic_strain += delta_lambda * dg_dsigma;

  } // end if plastic
}
//-----------------------------------------------------------------------------
PlasticityModel::~PlasticityModel()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
