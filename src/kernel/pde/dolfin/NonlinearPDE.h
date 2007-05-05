// Copyright (C) 2005-2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2007
//
// First added:  2005-10-24
// Last changed: 2007-05-05

#ifndef __NONLINEAR_PDE_H
#define __NONLINEAR_PDE_H

#include <dolfin/NonlinearProblem.h>
#include <dolfin/NewtonSolver.h>
#include <dolfin/Assembler.h>

namespace dolfin
{

  /// This class implements the solution functionality for nonlinear PDEs.
  
  class NonlinearPDE : public NonlinearProblem, public Parametrized
  {
  public:

    /// Constructor
    NonlinearPDE(Form& a, Form& L, Mesh& mesh, BoundaryCondition& bc);

    /// Constructor
    NonlinearPDE(Form& a, Form& L, Mesh& mesh, Array<BoundaryCondition*>& bcs);

    /// Destructor
    ~NonlinearPDE();
    
    /// User-defined function to compute F(u) its Jacobian
    void form(GenericMatrix& A, GenericVector& b, const GenericVector& x); 

    /// Solve PDE
    void solve(Function& u, real& t, const real& T, const real& dt);

  private:

    // The bilinear form
    Form& a;
    
    // The linear form
    Form& L;

    // The mesh
    Mesh& mesh;

    // The boundary conditions
    Array<BoundaryCondition*> bcs;

    // The solution vector
    Vector x;

    // Assembler 
    Assembler assembler;

    // Solver
    NewtonSolver newton_solver;
      
  };

}

#endif
