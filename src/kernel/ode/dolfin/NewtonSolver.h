// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEWTON_SOLVER_H
#define __NEWTON_SOLVER_H

#include <petsc/petscmat.h>
#include <petsc/petscvec.h>
#include <dolfin/constants.h>
#include <dolfin/NewGMRES.h>
#include <dolfin/NewJacobianMatrix.h>
#include <dolfin/NewVector.h>
#include <dolfin/TimeSlabSolver.h>

namespace dolfin
{

  class ODE;
  class NewTimeSlab;
  class NewMethod;

  /// This class implements Newton's method on time slabs. In each
  /// iteration, the system F(x) is evaluated at the current solution
  /// and then the linear system A dx = b is solved for the increment
  /// dx with A = F' the Jacobian of F and b = -F(x)

  class NewtonSolver : public TimeSlabSolver
  {
  public:

    /// Constructor
    NewtonSolver(ODE& ode, NewTimeSlab& timeslab, const NewMethod& method);

    /// Destructor
    ~NewtonSolver();

    /// Solve system
    void solve();

  protected:

    /// Start iterations (optional)
    void start();
    
    // Make an iteration
    real iteration();

  private:

    // Evaluate b = -F(x) at current x
    void beval();

    real* f;             // Values of right-hand side at quadrature points
    NewJacobianMatrix A; // Jacobian of time slab system
    NewVector dx;        // Increment for Newton's method
    NewVector b;         // Right-hand side -F(x)
    NewGMRES solver;     // GMRES solver
    
  };

}

#endif
