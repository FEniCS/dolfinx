// Copyright (C) 2006 Kristian Oelgaard and Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-13

#include <dolfin/DruckerPrager.h>

using namespace dolfin;

DruckerPrager::DruckerPrager(real E, real nu, real friction_angle, 
                real dilatancy_angle, real cohesion, real hardening_parameter) 
              : PlasticityModel(E, nu), _effective_stress(0.0), 
                _hardening_parameter(hardening_parameter)
{
  dg_dsigma_mod.resize(6, false);

  alpha_friction  = (6.0*sin(friction_angle))/(3-sin(friction_angle));
  k_friction      = (6.0*cohesion*cos(friction_angle))/(3-sin(friction_angle));
  alpha_dilatancy = (6.0*sin(dilatancy_angle))/(3-sin(dilatancy_angle));

  A.resize(6,6,false);
  A.clear();
  A(0,0)=2; A(1,1)=2; A(2,2)=2; A(3,3)=6; A(4,4)=6; A(5,5)=6;
  A(0,1)=-1; A(0,2)=-1; A(1,0)=-1; A(1,2)=-1; A(2,0)=-1; A(2,1)=-1;    
}
//-----------------------------------------------------------------------------
real DruckerPrager::hardening_parameter(real const equivalent_plastic_strain) const
{
  return _hardening_parameter;
}
//-----------------------------------------------------------------------------
real DruckerPrager::f(const uBlasVector& stress, const real equivalent_plastic_strain)
{

  return effective_stress(stress) + alpha_friction/3.0*(stress(0) + stress(1) + stress(2))
         - k_friction - _hardening_parameter * equivalent_plastic_strain;
}
//-----------------------------------------------------------------------------
void DruckerPrager::df(uBlasVector& df_dsigma, const uBlasVector& stress)
{
  _effective_stress = effective_stress(stress);
  df_dsigma(0) = (2*stress(0)-stress(1)-stress(2))/(2*_effective_stress) + alpha_friction/3.0;
  df_dsigma(1) = (2*stress(1)-stress(0)-stress(2))/(2*_effective_stress) + alpha_friction/3.0;
  df_dsigma(2) = (2*stress(2)-stress(0)-stress(1))/(2*_effective_stress) + alpha_friction/3.0;
  df_dsigma(3) = 6*stress(3)/(2*_effective_stress); 
  df_dsigma(4) = 6*stress(4)/(2*_effective_stress); 
  df_dsigma(5) = 6*stress(5)/(2*_effective_stress);
}
//-----------------------------------------------------------------------------
void DruckerPrager::dg(uBlasVector& dg_dsigma, const uBlasVector& stress)
{
  _effective_stress = effective_stress(stress);
  dg_dsigma(0) = (2*stress(0)-stress(1)-stress(2))/(2*_effective_stress) + alpha_dilatancy/3.0;
  dg_dsigma(1) = (2*stress(1)-stress(0)-stress(2))/(2*_effective_stress) + alpha_dilatancy/3.0;
  dg_dsigma(2) = (2*stress(2)-stress(0)-stress(1))/(2*_effective_stress) + alpha_dilatancy/3.0;
  dg_dsigma(3) = 6*stress(3)/(2*_effective_stress);
  dg_dsigma(4) = 6*stress(4)/(2*_effective_stress); 
  dg_dsigma(5) = 6*stress(5)/(2*_effective_stress);
}
//-----------------------------------------------------------------------------
void DruckerPrager::ddg(uBlasDenseMatrix& ddg_ddsigma, const uBlasVector& stress)
{
  _effective_stress = effective_stress(stress);
  dg_mod(dg_dsigma_mod, stress);
  ddg_ddsigma.assign(A/(2*_effective_stress) - outer_prod(dg_dsigma_mod, dg_dsigma_mod)/_effective_stress);
}
//-----------------------------------------------------------------------------
real DruckerPrager::effective_stress(const uBlasVector& stress)
{
  return sqrt(pow((stress(0) + stress(1) + stress(2)),2)-3*(stress(0) * stress(1) 
          + stress(0) * stress(2) + stress(1) * stress(2) -pow  (stress(3),2) 
          - pow(stress(4),2) -pow(stress(5),2)));
}
//-----------------------------------------------------------------------------
void DruckerPrager::dg_mod(uBlasVector& dg_dsigma_mod, const uBlasVector& stress)
{
  _effective_stress = effective_stress(stress);
  dg_dsigma_mod(0) = (2*stress(0)-stress(1)-stress(2))/(2*_effective_stress);
  dg_dsigma_mod(1) = (2*stress(1)-stress(0)-stress(2))/(2*_effective_stress);
  dg_dsigma_mod(2) = (2*stress(2)-stress(0)-stress(1))/(2*_effective_stress);
  dg_dsigma_mod(3) = 6*stress(3)/(2*_effective_stress);
  dg_dsigma_mod(4) = 6*stress(4)/(2*_effective_stress); 
  dg_dsigma_mod(5) = 6*stress(5)/(2*_effective_stress);
}
//-----------------------------------------------------------------------------
