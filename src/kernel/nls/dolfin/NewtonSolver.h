// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-10-23
// Last changed: 2005

#ifndef __NEWTON_SOLVER_H
#define __NEWTON_SOLVER_H

#include <petscsnes.h>

#include <dolfin/constants.h>
#include <dolfin/NonlinearFunction.h>
#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>

namespace dolfin
{

  /// This class defines a Newton solver for equations of the form F(u) = 0.
  
  class NewtonSolver 
  {
  public:

    /// Initialise nonlinear solver
    NewtonSolver();

    /// Initialise nonlinear solver for a given nonlinear function
    NewtonSolver(NonlinearFunction& nonlinear_function);

    /// Destructor
    ~NewtonSolver();
  
    /// Solve nonlinear problem F(u) = 0. Necessary matrix and vectors will be
    /// allocated
    uint solve(NonlinearFunction& nonlinear_function, Vector& x);

    /// Solve nonlinear problem F(u) = 0 when necessary matrix and vectors
    /// have already been allocated
    uint solve();

    /// Set Newton solver parameters
    void setParameters();

    /// Initialise Newton solver
    void init(Matrix& A, Vector& b, Vector& x);

    /// Return Newton iteration number
    int getIteration(SNES snes);

    /// Set nonlinear solve type
    void setType(std::string solver_type);

    /// Set maximum number of Netwon iterations
    void setMaxiter(int maxiter);

    /// Set relative convergence tolerance: du_i / du_0 < rtol
    void setRtol(real rtol);

    /// Set successive convergence tolerance: du_i+1 / du_i < stol
    void setStol(real stol);

    /// Set absolute convergence tolerance: du_i < atol
    void setAtol(real atol);

    /// Return pointer to PETSc nonlinear solver
    SNES solver();

  private:

    /// Function passed to PETSc to form RHS vector F(u)
    static int formRHS(SNES snes, Vec x, Vec f, void* nlfunc);

    /// Function passed to PETSc to form Jacobian (stiffness matrix) F'(u) = dF(u)/du
    static int formJacobian(SNES snes, Vec x, Mat* AA, Mat* BB, MatStructure *flag, void* nlfunc);

    /// Function passed to PETSc to form RHS vector and Jacobian 
    static int formSystem(SNES snes, Vec x, Vec f, void* nlfunc);

    /// Dummy function passed to PETSc for computing Jacobian 
    static int formDummy(SNES snes, Vec x, Mat* AA, Mat* BB, MatStructure *flag, void* nlfunc);

    /// Monitor function for nonlinear solver 
    static int monitor(SNES snes, int iter, real fnorm, void* dummy);

    // Pointer to nonlinear function
    NonlinearFunction* _nonlinear_function;

    // Pointer to Jacobian matrix
    Matrix* _A;

    // Pointer to RHS vector
    Vector* _b;

    // Pointer to solution vector
    Vector* _x;

    // PETSc nonlinear solver pointer
    SNES snes;

  };

}

#endif
