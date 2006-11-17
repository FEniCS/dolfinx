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

  uBlasDenseMatrix A(6,6);
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
real DruckerPrager::f(const uBlasVector& current_stress, const real equivalent_plastic_strain)
{
//  _effective_stress = effective_stress(current_stress);

  return effective_stress(current_stress) + alpha_friction/3.0*(current_stress(0) + current_stress(1) + current_stress(2))
         - k_friction - _hardening_parameter * equivalent_plastic_strain;
}
//-----------------------------------------------------------------------------
void DruckerPrager::df(uBlasVector& df_dsigma, const uBlasVector& current_stress)
{
  _effective_stress = effective_stress(current_stress);
  df_dsigma(0) = (2*current_stress(0)-current_stress(1)-current_stress(2))/(2*_effective_stress) + alpha_friction/3.0;
  df_dsigma(1) = (2*current_stress(1)-current_stress(0)-current_stress(2))/(2*_effective_stress) + alpha_friction/3.0;
  df_dsigma(2) = (2*current_stress(2)-current_stress(0)-current_stress(1))/(2*_effective_stress) + alpha_friction/3.0;
  df_dsigma(3) = 6*current_stress(3)/(2*_effective_stress); 
  df_dsigma(4) = 6*current_stress(4)/(2*_effective_stress); 
  df_dsigma(5) = 6*current_stress(5)/(2*_effective_stress);
}
//-----------------------------------------------------------------------------
void DruckerPrager::dg(uBlasVector& dg_dsigma, const uBlasVector& current_stress)
{
  _effective_stress = effective_stress(current_stress);
  dg_dsigma(0) = (2*current_stress(0)-current_stress(1)-current_stress(2))/(2*_effective_stress) + alpha_dilatancy/3.0;
  dg_dsigma(1) = (2*current_stress(1)-current_stress(0)-current_stress(2))/(2*_effective_stress) + alpha_dilatancy/3.0;
  dg_dsigma(2) = (2*current_stress(2)-current_stress(0)-current_stress(1))/(2*_effective_stress) + alpha_dilatancy/3.0;
  dg_dsigma(3) = 6*current_stress(3)/(2*_effective_stress);
  dg_dsigma(4) = 6*current_stress(4)/(2*_effective_stress); 
  dg_dsigma(5) = 6*current_stress(5)/(2*_effective_stress);
}
//-----------------------------------------------------------------------------
void DruckerPrager::ddg(uBlasDenseMatrix& ddg_ddsigma, const uBlasVector& current_stress)
{
  _effective_stress = effective_stress(current_stress);
  dg_mod(dg_dsigma_mod, current_stress);
  ddg_ddsigma.assign(Am/(2*_effective_stress) - outer_prod(dg_dsigma_mod, dg_dsigma_mod)/_effective_stress);
}
//-----------------------------------------------------------------------------
real DruckerPrager::effective_stress(const uBlasVector& current_stress)
{
  return sqrt(pow((current_stress(0) + current_stress(1) + current_stress(2)),2)-3*(current_stress(0) * current_stress(1) 
          + current_stress(0) * current_stress(2) + current_stress(1) * current_stress(2) -pow  (current_stress(3),2) 
          - pow(current_stress(4),2) -pow(current_stress(5),2)));
}
//-----------------------------------------------------------------------------
void DruckerPrager::dg_mod(uBlasVector& dg_dsigma_mod, const uBlasVector& current_stress)
{
  _effective_stress = effective_stress(current_stress);
  dg_dsigma_mod(0) = (2*current_stress(0)-current_stress(1)-current_stress(2))/(2*_effective_stress);
  dg_dsigma_mod(1) = (2*current_stress(1)-current_stress(0)-current_stress(2))/(2*_effective_stress);
  dg_dsigma_mod(2) = (2*current_stress(2)-current_stress(0)-current_stress(1))/(2*_effective_stress);
  dg_dsigma_mod(3) = 6*current_stress(3)/(2*_effective_stress);
  dg_dsigma_mod(4) = 6*current_stress(4)/(2*_effective_stress); 
  dg_dsigma_mod(5) = 6*current_stress(5)/(2*_effective_stress);
}
//-----------------------------------------------------------------------------
