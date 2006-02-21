// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-10-24
// Last changed: 2005-12-05

#ifndef __NONLINEAR_PDE_H
#define __NONLINEAR_PDE_H


#include <dolfin/NonlinearFunction.h>

namespace dolfin
{
  class BilinearForm; 
  class BoundaryCondition;
  class Function;
  class LinearForm;
  class Matrix;
  class Mesh;
  class Vector;

  /// This class acts as a base class for nonlinear PDE's and 
  /// the nonlinear function F(u) and its Jacobian J = dF(u)/du.
  
  class NonlinearPDE : public NonlinearFunction
  {
  public:

    /// Create nonlinear function
    NonlinearPDE();

    /// Create nonlinear PDE with natural  boundary conditions
    NonlinearPDE(BilinearForm& a, LinearForm& L, Mesh& mesh);

    /// Create nonlinear PDE with Dirichlet boundary conditions
    NonlinearPDE(BilinearForm& a, LinearForm& L, Mesh& mesh,
              BoundaryCondition& bc);

    /// Destructor
    virtual ~NonlinearPDE();

     /// User-defined function to compute F(u) its Jacobian
    virtual void form(Matrix& A, Vector& b, const Vector& x);

     /// User-defined function to compute F(u)
    virtual void F(Vector& b, const Vector& x);

     /// User-defined function to compute Jacobian matrix
    virtual void J(Matrix& A, const Vector& x);

     /// Solve nonlinear PDE
    uint solve(Function& u);

     /// Solve nonlinear PDE
    Function solve();

    /// Friends
    friend class NewtonSolver;


  protected:

    BilinearForm* _a;
    LinearForm* _Lf;
    Mesh* _mesh;
    BoundaryCondition* _bc;

  };
}

#endif
