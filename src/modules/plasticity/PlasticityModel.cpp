// Copyright (C) 2006 Kristian Oelgaard and Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-13

#include <dolfin/PlasticityModel.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
PlasticityModel::PlasticityModel(real E, real nu) : _hardening_parameter(0.0)
{
  //  Lame coefficients
  real lam = nu*E/((1+nu)*(1-2*nu));
  real mu = E/(2*(1+nu));

  // Elastic tangent
  elastic_tangent = C_m(lam, mu);
}
//-----------------------------------------------------------------------------
PlasticityModel::~PlasticityModel()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
real PlasticityModel::hardening_parameter(real const equivalent_plastic_strain) const 
{
  return _hardening_parameter;
}
//-----------------------------------------------------------------------------
real PlasticityModel::kappa(real equivalent_plastic_strain, 
                const uBlasVector& current_stress, const real lambda_dot)
{
  equivalent_plastic_strain += lambda_dot;

  return equivalent_plastic_strain;
}
//-----------------------------------------------------------------------------
void PlasticityModel::dg(uBlasVector &dg_dsigma, const uBlasVector& current_stress)
{
  // If dg is not overloaded associative flow is assumed (dg/dsigma = df/dsigma)
  df(dg_dsigma, current_stress);
}
//-----------------------------------------------------------------------------
uBlasDenseMatrix PlasticityModel::C_m(real lam, real mu)
{
  uBlasDenseMatrix B(6,6);
  B.clear();

  B(0,0)=lam+2*mu, B(1,1)=lam+2*mu, B(2,2)=lam+2*mu;
  B(3,3)=mu, B(4,4)=mu, B(5,5)=mu;
  B(0,1)=lam, B(0,2)=lam, B(1,0)=lam;
  B(1,2)=lam, B(2,0)=lam, B(2,1)=lam;

  return B;
}
//-----------------------------------------------------------------------------
