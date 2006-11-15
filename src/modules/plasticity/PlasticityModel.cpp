// Copyright (C) 2006 Kristian Oelgaard and Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-13

#include <dolfin/PlasticityModel.h>

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
}
//-----------------------------------------------------------------------------
real PlasticityModel::hardening_parameter(real eps_eq)
{
  return _hardening_parameter;
}
//-----------------------------------------------------------------------------
real PlasticityModel::kappa(real equivalent_plastic_strain, 
                uBlasVector &sigma, real lambda_dot)
{
  equivalent_plastic_strain += lambda_dot;

  return equivalent_plastic_strain;
}
//-----------------------------------------------------------------------------
void PlasticityModel::dg(uBlasVector &dg_dsigma, uBlasVector& sigma)
{
  df(dg_dsigma, sigma);
}
//-----------------------------------------------------------------------------
