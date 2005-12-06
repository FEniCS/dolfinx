// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-10-23
// Last changed: 2005-12-05

#ifndef __NEWTON_SOLVER_H
#define __NEWTON_SOLVER_H

#include <dolfin/constants.h>
#include <dolfin/NonlinearFunction.h>
#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <dolfin/KrylovSolver.h>
#include <dolfin/BilinearForm.h>
#include <dolfin/LinearForm.h>
#include <dolfin/Mesh.h>
#include <dolfin/BoundaryCondition.h>

namespace dolfin
{

  /// This class defines a Newton solver for equations of the form F(u) = 0.
  
  class NewtonSolver : public KrylovSolver
  {
  public:

    //FIXME: implement methods other than plain Newton
    enum Method { newton };
     
    /// Initialise nonlinear solver
    NewtonSolver();

    /// Destructor
    ~NewtonSolver();
  
    /// Solve nonlinear problem F(u) = 0
    uint solve(BilinearForm& a, LinearForm& L, BoundaryCondition& bc, Mesh& mesh,  
        Vector& x);

    /// Solve nonlinear problem F(u) = 0 for given NonlinearFunction
    uint solve(NonlinearFunction& nonlinearfunction, Vector& x);

/*
    /// Set nonlinear solve type
    void setType(std::string solver_type);

*/

    /// Set maximum number of Netwon iterations
    void setNewtonMaxiter(uint maxiter) const;

    /// Set relative convergence tolerance: ||F_i|| / ||F_0|| < rtol
    void setNewtonRtol(real rtol) const;

    /// Set absolute convergence tolerance: ||F_i|| < atol
    void setNewtonAtol(real atol) const;

    /// Return Newton iteration number
    uint getIteration() const;

  private:

    // Type of Newton method
    Method method;

    // Number of Newton iterations
    uint iteration;

    // Total number of Krylov iterations
    uint kryloviterations;

    // Residuals
    real residual, relative_residual;

    // True if information should be printed at each iteration
    bool report;

    // Jacobian matrix
    Matrix A;

    // Resdiual vector
    Vector b;

    // Solution vector
    Vector dx;

  };

}

#endif
