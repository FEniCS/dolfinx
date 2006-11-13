// Copyright (C) 2006 Kristian Oelgaard and Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-13

#ifndef __VON_MISE_H
#define __VON_MISE_H

#include "PlasticityModel.h"
namespace dolfin
{

class Von_Mise : public PlasticityModel
{
  public:

    Von_Mise(double &yield_stress, double &hardening_parameter) : PlasticityModel(), sig_o(yield_stress),  H(hardening_parameter)
    {
      Am = A_m();
      dg_dsigma.resize(6, false);
    }

    void effective_stress(double &sig_e, ublas::vector<double> &sig)
    {
      sig_e = sqrt(pow((sig(0) + sig(1) + sig(2)),2)-3*(sig(0) * sig(1) + sig(0) * sig(2) + sig(1) * sig(2) -pow(sig(3),2) -pow(sig(4),2) -pow(sig(5),2)));
    }

    double effective_stress()
    {
      return sig_e; 
    }

    void df(ublas::vector<double> &a, ublas::vector<double> &sig)
    {
      a(0) = (2*sig(0)-sig(1)-sig(2))/(2*sig_e);
      a(1) = (2*sig(1)-sig(0)-sig(2))/(2*sig_e);
      a(2) = (2*sig(2)-sig(0)-sig(1))/(2*sig_e);
      a(3) = 6*sig(3)/(2*sig_e);
      a(4) = 6*sig(4)/(2*sig_e);
      a(5) = 6*sig(5)/(2*sig_e);
    }

    ublas::matrix <double> A_m()
    {
      ublas::matrix <double> A(6,6);
      A.clear(); 
      A(0,0)=2, A(1,1)=2, A(2,2)=2, A(3,3)=6, A(4,4)=6, A(5,5)=6;
      A(0,1)=-1, A(0,2)=-1, A(1,0)=-1, A(1,2)=-1, A(2,0)=-1, A(2,1)=-1;    

      return A;
    }

    void ddg(ublas::matrix<double> &ddg_ddsigma, ublas::vector<double> &sig)
    {
      df(dg_dsigma,sig);
      ddg_ddsigma.assign(Am/(2*sig_e) - outer_prod(dg_dsigma,dg_dsigma)/sig_e);
    }

    void f(double &f0, ublas::vector<double> &sig, double &eps_eq)
    {
      effective_stress(sig_e, sig);
      f0 =  sig_e - sig_o - H * eps_eq;
    }

    double hardening_parameter(double &eps_eq)
    {
      return H;
    }

  private:

    ublas::matrix<double> Am;
    ublas::vector<double> dg_dsigma;

    double sig_o, sig_e, H;
};

}
#endif
