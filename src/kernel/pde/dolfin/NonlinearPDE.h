// Copyright (C) 2005-2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-10-24
// Last changed: 2006-05-07

#ifndef __NONLINEAR_PDE_H
#define __NONLINEAR_PDE_H

#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <dolfin/GenericPDE.h>
#include <dolfin/NonlinearProblem.h>
#include <dolfin/NewtonSolver.h>

namespace dolfin
{
  class BilinearForm; 
  class BoundaryCondition;
  class Function;
  class LinearForm;
  class Mesh;

  /// This class acts as a base class for nonlinear PDE's and 
  /// the nonlinear function F(u) and its Jacobian J = dF(u)/du.
  
  class NonlinearPDE : public GenericPDE, public NonlinearProblem
  {
  public:

    /// Create nonlinear PDE with natural  boundary conditions
    NonlinearPDE(BilinearForm& a, LinearForm& L, Mesh& mesh);

    /// Create nonlinear PDE with Dirichlet boundary conditions
    NonlinearPDE(BilinearForm& a, LinearForm& L, Mesh& mesh,
              BoundaryCondition& bc);

    /// Destructor
    virtual ~NonlinearPDE();

     /// User-defined function to compute F(u) its Jacobian
    virtual void form(GenericMatrix& A, GenericVector& b, const GenericVector& x);

    /// Solve PDE (in general a mixed system). If the function u has been 
    /// initialised, it is used as a starting value.
    uint solve(Function& u);

//     /// User-defined function to compute F(u)
//    virtual void F(GenericVector& b, const GenericVector& x);

//     /// User-defined function to compute Jacobian matrix
//    virtual void J(GenericMatrix& A, const GenericVector& x);

    /// Return the element dimension
    uint elementdim();

    /// Return the bilinear form a(.,.)
    BilinearForm& a();

    /// Return the linear form L(.,.)
    LinearForm& L();

    /// Return the mesh
    Mesh& mesh();

    /// Return the boundary condition
    BoundaryCondition& bc();

  protected:

    BilinearForm* _a;
    LinearForm* _Lf;
    Mesh* _mesh;
    BoundaryCondition* _bc;

  private:

    NewtonSolver newton_solver;

  };
}

#endif
