// Copyright (C) 2006 Kristian Oelgaard and Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-13

#include "dolfin/DruckerPrager.h"

using namespace dolfin;

DruckerPrager::DruckerPrager(real& _phi, real& _psi, real& _c, real _H) 
            : PlasticityModel(), phi(_phi), psi(_psi), c(_c), sig_e(0.0), H(_H)
{
  Am = A_m();
  dg_dsigma_mod.resize(6, false);

  alpha_f = (6.0*sin(phi))/(3-sin(phi));
  k_f = (6.0*c*cos(phi))/(3-sin(phi));
  alpha_g = (6.0*sin(psi))/(3-sin(psi));
}
//-----------------------------------------------------------------------------
void DruckerPrager::f(real& f0, uBlasVector& sig, real& eps_eq)
{
  sig_e = effective_stress(sig);
  f0 =  sig_e + alpha_f/3.0*(sig(0) + sig(1) + sig(2)) -k_f- H*eps_eq ;
}
//-----------------------------------------------------------------------------
real DruckerPrager::effective_stress(uBlasVector& sig)
{
  return sqrt(pow((sig(0) + sig(1) + sig(2)),2)-3*(sig(0) * sig(1) 
              + sig(0) * sig(2) + sig(1) * sig(2) -pow(sig(3),2) 
              -pow(sig(4),2) -pow(sig(5),2)));
}
//-----------------------------------------------------------------------------
real DruckerPrager::hardening_parameter(real& eps_eq)
{
  return H;
}
//-----------------------------------------------------------------------------
real DruckerPrager::effective_stress()
{
  return sig_e; 
}
//-----------------------------------------------------------------------------
void DruckerPrager::df(uBlasVector& a, uBlasVector& sig)
{
  a(0) = (2*sig(0)-sig(1)-sig(2))/(2*sig_e) + alpha_f/3.0;
  a(1) = (2*sig(1)-sig(0)-sig(2))/(2*sig_e) + alpha_f/3.0;
  a(2) = (2*sig(2)-sig(0)-sig(1))/(2*sig_e) + alpha_f/3.0;
  a(3) = 6*sig(3)/(2*sig_e); 
  a(4) = 6*sig(4)/(2*sig_e); 
  a(5) = 6*sig(5)/(2*sig_e);
}
//-----------------------------------------------------------------------------
void DruckerPrager::dg(uBlasVector& b, uBlasVector& sig)
{
  b(0) = (2*sig(0)-sig(1)-sig(2))/(2*sig_e) + alpha_g/3.0;
  b(1) = (2*sig(1)-sig(0)-sig(2))/(2*sig_e) + alpha_g/3.0;
  b(2) = (2*sig(2)-sig(0)-sig(1))/(2*sig_e) + alpha_g/3.0;
  b(3) = 6*sig(3)/(2*sig_e);
  b(4) = 6*sig(4)/(2*sig_e); 
  b(5) = 6*sig(5)/(2*sig_e);
}
//-----------------------------------------------------------------------------
void DruckerPrager::dg_mod(uBlasVector& b_mod, uBlasVector& sig, real sig_e)
{
  b_mod(0) = (2*sig(0)-sig(1)-sig(2))/(2*sig_e);
  b_mod(1) = (2*sig(1)-sig(0)-sig(2))/(2*sig_e);
  b_mod(2) = (2*sig(2)-sig(0)-sig(1))/(2*sig_e);
  b_mod(3) = 6*sig(3)/(2*sig_e);
  b_mod(4) = 6*sig(4)/(2*sig_e); 
  b_mod(5) = 6*sig(5)/(2*sig_e);
}
//-----------------------------------------------------------------------------
uBlasDenseMatrix DruckerPrager::A_m()
{
  uBlasDenseMatrix A(6,6);
  A.clear(); 
  A(0,0)=2; A(1,1)=2; A(2,2)=2; A(3,3)=6; A(4,4)=6; A(5,5)=6;
  A(0,1)=-1; A(0,2)=-1; A(1,0)=-1; A(1,2)=-1; A(2,0)=-1; A(2,1)=-1;    

  return A;
}
//-----------------------------------------------------------------------------
void DruckerPrager::ddg(uBlasDenseMatrix& ddg_ddsigma, uBlasVector& sig)
{
  sig_e = effective_stress(sig);
  dg_mod(dg_dsigma_mod, sig, sig_e);
  ddg_ddsigma.assign(Am/(2*sig_e) - outer_prod(dg_dsigma_mod,dg_dsigma_mod)/sig_e);
}
//-----------------------------------------------------------------------------
