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

  A.resize(6,6,false);
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
real VonMises::f(const uBlasVector& stress, const real equivalent_plastic_strain)
{
  _effective_stress = effective_stress(stress);

  return _effective_stress - _yield_stress - _hardening_parameter*equivalent_plastic_strain;
}
//-----------------------------------------------------------------------------
void VonMises::df(uBlasVector& df_dsigma, const uBlasVector& stress)
{
  _effective_stress = effective_stress(stress);

  df_dsigma(0) = (2*stress(0)-stress(1)-stress(2))/(2*_effective_stress);
  df_dsigma(1) = (2*stress(1)-stress(0)-stress(2))/(2*_effective_stress);
  df_dsigma(2) = (2*stress(2)-stress(0)-stress(1))/(2*_effective_stress);
  df_dsigma(3) = 6*stress(3)/(2*_effective_stress);
  df_dsigma(4) = 6*stress(4)/(2*_effective_stress);
  df_dsigma(5) = 6*stress(5)/(2*_effective_stress);
}
//-----------------------------------------------------------------------------
void VonMises::ddg(uBlasDenseMatrix& ddg_ddsigma, const uBlasVector& stress)
{
  _effective_stress = effective_stress(stress);
  df(dg_dsigma, stress);
  ddg_ddsigma.assign(A/(2*_effective_stress) - outer_prod(dg_dsigma,dg_dsigma)/_effective_stress);
}
//-----------------------------------------------------------------------------
real VonMises::effective_stress(const uBlasVector& stress)
{
  return sqrt(pow((stress(0) + stress(1) 
          + stress(2)),2) - 3*(stress(0)*stress(1) 
          + stress(0)*stress(2) 
          + stress(1)*stress(2) -pow(stress(3),2) 
          - pow(stress(4),2) -pow(stress(5),2)));
}
//-----------------------------------------------------------------------------
