// Copyright (C) 2006 Kristian Oelgaard and Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-13

#include <dolfin/VonMises.h>

using namespace dolfin;

VonMises::VonMises(real E, real nu, real yield_stress, real hardening_parameter) 
                 : PlasticityModel(E, nu), _yield_stress(yield_stress), 
                   _effective_stress(0.0), _hardening_parameter(hardening_parameter)
{
  dg_dsigma.resize(6, false);

  uBlasDenseMatrix A(6,6);
  A.clear(); 
  A(0,0) = 2, A(1,1) = 2, A(2,2) = 2, A(3,3) = 6, A(4,4) = 6, A(5,5) = 6;
  A(0,1) = -1, A(0,2) = -1, A(1,0) = -1, A(1,2) = -1, A(2,0) = -1, A(2,1)=-1;    
}
//-----------------------------------------------------------------------------
real VonMises::hardening_parameter(const real equivalent_plastic_strain) const
{
  return _hardening_parameter;
}
//-----------------------------------------------------------------------------
real VonMises::f(const uBlasVector& current_stress, const real equivalent_plastic_strain)
{
  _effective_stress = effective_stress(current_stress);

  return _effective_stress - _yield_stress - _hardening_parameter*equivalent_plastic_strain;
}
//-----------------------------------------------------------------------------
void VonMises::df(uBlasVector& df_dsigma, const uBlasVector& current_stress)
{
  _effective_stress = effective_stress(current_stress);

  df_dsigma(0) = (2*current_stress(0)-current_stress(1)-current_stress(2))/(2*_effective_stress);
  df_dsigma(1) = (2*current_stress(1)-current_stress(0)-current_stress(2))/(2*_effective_stress);
  df_dsigma(2) = (2*current_stress(2)-current_stress(0)-current_stress(1))/(2*_effective_stress);
  df_dsigma(3) = 6*current_stress(3)/(2*_effective_stress);
  df_dsigma(4) = 6*current_stress(4)/(2*_effective_stress);
  df_dsigma(5) = 6*current_stress(5)/(2*_effective_stress);
}
//-----------------------------------------------------------------------------
void VonMises::ddg(uBlasDenseMatrix& ddg_ddsigma, const uBlasVector& current_stress)
{
  _effective_stress = effective_stress(current_stress);
  df(dg_dsigma, current_stress);
  ddg_ddsigma.assign(Am/(2*_effective_stress) - outer_prod(dg_dsigma,dg_dsigma)/_effective_stress);
}
//-----------------------------------------------------------------------------
real VonMises::effective_stress(const uBlasVector& current_stress)
{
  return sqrt(pow((current_stress(0) + current_stress(1) 
          + current_stress(2)),2) - 3*(current_stress(0)*current_stress(1) 
          + current_stress(0)*current_stress(2) 
          + current_stress(1)*current_stress(2) -pow(current_stress(3),2) 
          - pow(current_stress(4),2) -pow(current_stress(5),2)));
}
//-----------------------------------------------------------------------------
