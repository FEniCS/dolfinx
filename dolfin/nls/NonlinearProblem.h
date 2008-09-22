// Copyright (C) 2005-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-10-24
// Last changed: 2008-08-26

#ifndef __NONLINEAR_PROBLEM_H
#define __NONLINEAR_PROBLEM_H

namespace dolfin
{

  class GenericMatrix;
  class GenericVector;

  /// This is a base class for nonlinear problems which can return the 
  /// nonlinear function F(u) and its Jacobian J = dF(u)/du.
  
  class NonlinearProblem
  {
  public:

    /// Create nonlinear problem
    NonlinearProblem();

    /// Destructor
    virtual ~NonlinearProblem();

     /// User-defined function to compute F(u) its Jacobian
    virtual void form(GenericMatrix& A, GenericVector& b, const GenericVector& x);

     /// User-defined function to compute F(u)
    virtual void F(GenericVector& b, const GenericVector& x);

     /// User-defined function to compute Jacobian matrix
    virtual void J(GenericMatrix& A, const GenericVector& x);

  };
}

#endif
