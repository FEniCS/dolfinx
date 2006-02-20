// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-10-23
// Last changed: 2006-02-20

#ifndef __NEWTON_SOLVER_H
#define __NEWTON_SOLVER_H

#include <dolfin/NonlinearFunction.h>
#include <dolfin/NonlinearPDE.h>
#include <dolfin/KrylovSolver.h>
#include <dolfin/BilinearForm.h>

namespace dolfin
{
  class BoundaryCondition;
  class LinearForm;
  class Matrix;
  class Mesh;
  class Vector;

  /// This class defines a Newton solver for equations of the form F(u) = 0.
  
  class NewtonSolver : public KrylovSolver
  {
  public:

    //FIXME: implement methods other than plain Newton (modified, line search, etc.)
    enum Method { newton };
     
    /// Initialise nonlinear solver
    NewtonSolver();

    /// Destructor
    ~NewtonSolver();
  
    /// Solve nonlinear PDE
    uint solve(BilinearForm& a, LinearForm& L, BoundaryCondition& bc, Mesh& mesh,
        Function& u);

    /// Solve nonlinear PDE
    uint solve(NonlinearPDE& nonlinearfunction, Function& u);

    /// Solve abstract nonlinear problem F(x) = 0 for given vector F and 
    /// Jacobian dF/dx
    uint solve(NonlinearFunction& nonlinearfunction, Vector& x);

    /// Return Newton iteration number
    uint getIteration() const;

  private:

    // Type of Newton method
    Method method;

    // Number of Newton iterations
    uint iteration;

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
