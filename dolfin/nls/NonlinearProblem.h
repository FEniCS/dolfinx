// Copyright (C) 2005-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
//
// First added:  2005-10-24
// Last changed: 2008-08-26

#ifndef __NONLINEAR_PROBLEM_H
#define __NONLINEAR_PROBLEM_H

#include <dolfin/log/log.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/GenericVector.h>

namespace dolfin
{

  // Forward declarations
  //class GenericMatrix;
  //class GenericVector;

  /// This is a base class for nonlinear problems which can return the
  /// nonlinear function F(u) and its Jacobian J = dF(u)/du.

  class NonlinearProblem
  {
  public:

    /// Constructor
    NonlinearProblem();

    /// Destructor
    virtual ~NonlinearProblem();

    /// Function called by Newton solver before requesting F or J. 
    /// This can be used to comoute F and J together 
    virtual void form(GenericMatrix& A, GenericVector& b, const GenericVector& x) 
      { /* Do nothing if not supplied by the user */ };

    /// Compute F at current point x
    virtual void F(GenericVector& b, const GenericVector& x)
    { error("F not provided"); }

    /// Compute J = F' at current point x
    virtual void J(GenericMatrix& A, const GenericVector& x)
    { error("J not provided"); }

    // For testing
    virtual void test_F(unsigned int a, unsigned int b) { error("test F not provided"); }
    virtual void test_J(unsigned int a, unsigned int b) { error("test J not provided"); }

  };
}

#endif
