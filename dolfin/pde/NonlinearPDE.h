// Copyright (C) 2005-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2007
//
// First added:  2005-10-24
// Last changed: 2008-09-03

#ifndef __NONLINEAR_PDE_H
#define __NONLINEAR_PDE_H

#include <dolfin/nls/NonlinearProblem.h>
#include <dolfin/nls/NewtonSolver.h>
#include <dolfin/fem/Assembler.h>
#include <dolfin/la/Vector.h>

namespace dolfin
{
  // Forward declarations
  class Form;
  class Mesh;
  class DirichletBC;
  class GenericMatrix;
  class GenericVector;

  /// This class provides automated solution of nonlinear PDEs.
  
  class NonlinearPDE : public NonlinearProblem, public Parametrized
  {
  public:

    /// Constructor
    NonlinearPDE(Form& a, Form& L, Mesh& mesh, DirichletBC& bc);

    /// Constructor
    NonlinearPDE(Form& a, Form& L, Mesh& mesh, Array<DirichletBC*>& bcs);

    /// Destructor
    ~NonlinearPDE();

    /// Function called before Jacobian matrix and RHS vector are formed. Users
    /// can supply this function to perform updates.
    virtual void update(const GenericVector& x);

    /// Compute F(u)
    void F(GenericVector& b, const GenericVector& x); 

    /// Compute Jacobian of F(u)
    void J(GenericMatrix& A, const GenericVector& x); 

    /// Solve PDE
    void solve(Function& u, double& t, const double& T, const double& dt);

  private:

    // The bilinear form
    Form& a;
    
    // The linear form
    Form& L;

    // The mesh
    Mesh& mesh;

    // The boundary conditions
    Array<DirichletBC*> bcs;

    // The solution vector
    Vector x;

    // Assembler 
    Assembler assembler;

    // Solver
    NewtonSolver newton_solver;
      
  };

}

#endif
