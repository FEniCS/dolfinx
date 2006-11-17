// Copyright (C) 2006 Kristian Oelgaard and Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-13

#include <dolfin/PlasticityModel.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
PlasticityModel::PlasticityModel(const real E, const real nu) : _hardening_parameter(0.0)
{
  // Lame coefficients
  real lambda = nu*E/((1+nu)*(1-2*nu));
  real mu = E/(2*(1+nu));

  // Create elastic tangent
  uBlasDenseMatrix elastic_tangent(6,6);
  elastic_tangent.clear();
  elastic_tangent(0,0) = lambda+2*mu;
  elastic_tangent(1,1) = lambda+2*mu; 
  elastic_tangent(2,2) = lambda+2*mu;
  elastic_tangent(3,3) = mu, elastic_tangent(4,4) = mu, elastic_tangent(5,5) = mu;
  elastic_tangent(0,1) = lambda, elastic_tangent(0,2) = lambda, elastic_tangent(1,0) = lambda;
  elastic_tangent(1,2) = lambda, elastic_tangent(2,0) = lambda, elastic_tangent(2,1) = lambda;
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
                const uBlasVector& current_stress, const real lambda_dot) const
{
  return equivalent_plastic_strain += lambda_dot;
}
//-----------------------------------------------------------------------------
void PlasticityModel::dg(uBlasVector &dg_dsigma, const uBlasVector& current_stress)
{
  // Assume associative flow (dg/dsigma = df/dsigma)
  df(dg_dsigma, current_stress);
}
//-----------------------------------------------------------------------------
