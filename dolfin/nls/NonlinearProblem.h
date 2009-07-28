// Copyright (C) 2005-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-10-24
// Last changed: 2008-08-26
//
// Modified by Anders Logg, 2008.

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

    /// Constructor
    NonlinearProblem() {}

    /// Destructor
    virtual ~NonlinearProblem() {}

    /// Function called by Newton solver before requesting F or J. 
    /// This can be used when it is used to comoute F and J together 
    virtual void form(GenericMatrix& A, GenericVector& b, const GenericVector& x) 
    { /* Do nothing if not supplied bu the user */ };

    /// Compute F at current point x
    virtual void F(GenericVector& b, const GenericVector& x) = 0;

    /// Compute J = F' at current point x
    virtual void J(GenericMatrix& A, const GenericVector& x) = 0;

  };
}

#endif
