// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-10-24
// Last changed: 2006-02-24

#ifndef __NONLINEAR_PROBLEM_H
#define __NONLINEAR_PROBLEM_H

namespace dolfin
{
  class BilinearForm; 
  class BoundaryCondition;
  class LinearForm;
  class Matrix;
  class Mesh;
  class Vector;

  /// This class acts as a base class for nonlinear problems which can return 
  /// the nonlinear function F(u) and its Jacobian J = dF(u)/du.
  
  class NonlinearProblem
  {
  public:

    /// Create nonlinear problem
    NonlinearProblem();

    /// Destructor
    virtual ~NonlinearProblem();

     /// User-defined function to compute F(u) its Jacobian
    virtual void form(Matrix& A, Vector& b, const Vector& x);

     /// User-defined function to compute F(u)
    virtual void F(Vector& b, const Vector& x);

     /// User-defined function to compute Jacobian matrix
    virtual void J(Matrix& A, const Vector& x);

  };
}

#endif
