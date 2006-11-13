// Copyright (C) 2006 Kristian Oelgaard and Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-13

#ifndef __PLASTICITY_MODEL_H
#define __PLASTICITY_MODEL_H

#include <dolfin/DenseMatrix.h>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/io.hpp>

namespace dolfin
{
  class PlasticityModel
  {
  public:

    /// Constructor
    PlasticityModel();
    
    /// Destructor
    virtual ~PlasticityModel();

    /// Initialising variables
    void initialise();    

    /// Hardening parameter A
    virtual double hardening_parameter(double &eps_eq);

    /// Value of yield function, f
    virtual void f(double &f0, ublas::vector<double> &sig, double &eps_eq) = 0;

    /// first derivative of f with respect to sigma, a
    virtual void df(ublas::vector <double> &a, ublas::vector <double> &sig) = 0;

    /// first derivative of g with respect to sigma, b
    virtual void dg(ublas::vector <double> &b, ublas::vector <double> &sig);
    
    /// Second derivative of g with respect to sigma, db_dsig
    virtual void ddg(ublas::matrix <double> &dm_dsig, ublas::vector <double> &sig) = 0;

    /// Equivalent plastic strain, kappa
    virtual void kappa(double &eps_eq, ublas::vector<double> &sig, double &lambda);
    
    /// Backward Euler return mapping
    void return_mapping(ublas::matrix <double> &cons_t, ublas::matrix <double> &D, ublas::vector <double> &t_sig, ublas::vector <double> &eps_p, double &eps_eq);

  private:
    ublas::matrix<double> R;
    ublas::matrix<double> ddg_ddsigma;

    DenseMatrix inverse_Q;
    ublas::identity_matrix <double> I;

    ublas::vector<double> df_dsigma, dg_dsigma, sigma_current, sigma_dot, sigma_residual, Rm, Rn, RinvQ;

    double residual_f, _hardening_parameter, delta_lambda, lambda_dot;
  
  };
}

#endif
