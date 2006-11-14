// Copyright (C) 2006 Kristian Oelgaard and Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-13

#include "dolfin/VonMises.h"


using namespace dolfin;

VonMises::VonMises(real& yield_stress, real& hardening_parameter) 
          : PlasticityModel(), sig_o(yield_stress),  H(hardening_parameter)
{
  Am = A_m();
  dg_dsigma.resize(6, false);
}
//-----------------------------------------------------------------------------
void VonMises::effective_stress(real& sig_e, uBlasVector& sig)
{
  sig_e = sqrt(pow((sig(0) + sig(1) + sig(2)),2)-3*(sig(0) * sig(1) 
          + sig(0) * sig(2) + sig(1) * sig(2) -pow  (sig(3),2) 
          - pow(sig(4),2) -pow(sig(5),2)));
}
//-----------------------------------------------------------------------------
real VonMises::effective_stress()
{
  return sig_e; 
}
//-----------------------------------------------------------------------------
void VonMises::df(uBlasVector& a, uBlasVector& sig)
{
  a(0) = (2*sig(0)-sig(1)-sig(2))/(2*sig_e);
  a(1) = (2*sig(1)-sig(0)-sig(2))/(2*sig_e);
  a(2) = (2*sig(2)-sig(0)-sig(1))/(2*sig_e);
  a(3) = 6*sig(3)/(2*sig_e);
  a(4) = 6*sig(4)/(2*sig_e);
  a(5) = 6*sig(5)/(2*sig_e);
}
//-----------------------------------------------------------------------------
uBlasDenseMatrix VonMises::A_m()
{
  uBlasDenseMatrix A(6,6);
  A.clear(); 
  A(0,0)=2, A(1,1)=2, A(2,2)=2, A(3,3)=6, A(4,4)=6, A(5,5)=6;
  A(0,1)=-1, A(0,2)=-1, A(1,0)=-1, A(1,2)=-1, A(2,0)=-1, A(2,1)=-1;    

  return A;
}
//-----------------------------------------------------------------------------
void VonMises::ddg(uBlasDenseMatrix& ddg_ddsigma, uBlasVector& sig)
{
  df(dg_dsigma,sig);
  ddg_ddsigma.assign(Am/(2*sig_e) - outer_prod(dg_dsigma,dg_dsigma)/sig_e);
}
//-----------------------------------------------------------------------------
void VonMises::f(real& f0, uBlasVector& sig, real& eps_eq)
{
  effective_stress(sig_e, sig);
  f0 =  sig_e - sig_o - H * eps_eq;
}
//-----------------------------------------------------------------------------
real VonMises::hardening_parameter(real& eps_eq)
{
  return H;
}
//-----------------------------------------------------------------------------
