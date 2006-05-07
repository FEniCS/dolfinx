// Copyright (C) 2002-2006 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Jansson 2005.
//
// First added:  2002
// Last changed: 2006-05-07

#ifndef __HEAT_SOLVER_H
#define __HEAT_SOLVER_H

#ifdef HAVE_PETSC_H

#include <dolfin/Solver.h>
#include <dolfin/ODE.h>
#include <dolfin/TimeStepper.h>
#include <dolfin/Heat.h>

namespace dolfin
{

  /// This class implements a solver for Heat's equation.
  ///
  /// FIXME: Make dimension-independent (currently 2D)

  class HeatODE;

  class HeatSolver : public Solver
  {
  public:
    
    /// Create Heat solver
    HeatSolver(Mesh& mesh, Function& f, BoundaryCondition& bc, real& T);
    
    /// Solve Heat's equation
    void solve();

    // Compute f(u) in dot(u) = f(u)
    void fu();

    /// Solve Heat's equation (static version)
    static void solve(Mesh& mesh, Function& f, BoundaryCondition& bc, real& T);
  
    Mesh& mesh;
    Function& f;
    BoundaryCondition& bc;

    Matrix Dummy;
    Vector x, dotu, m;

    Heat::LinearForm::TestElement element;

    Function u;

    Heat::BilinearForm a;
    Heat::LinearForm L;

    uint N, fevals;

    // ODE

    HeatODE* ode;
    TimeStepper* ts;

    real T;

  private:

  };

  class HeatODE : public ODE
  {
  public:
    HeatODE(HeatSolver& solver);
    real u0(unsigned int i);

    /// Evaluate right-hand side (mono-adaptive version)
    virtual void f(const real u[], real t, real y[]);
    virtual bool update(const real u[], real t, bool end);

    void fromArray(const real u[], Vector& x, uint offset, uint size);
    void toArray(real y[], Vector&x, uint offset, uint size);

    HeatSolver& solver;
  };


}

#endif

#endif
