// Copyright (C) 2004-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2006.
//
// First added:  2004
// Last changed: 2006-02-24

#ifndef __TIMEDEPENDENT_PDE_H
#define __TIMEDEPENDENT_PDE_H

#ifdef HAVE_PETSC_H

#include <dolfin/GenericPDE.h>
#include <dolfin/ODE.h>

namespace dolfin
{

  class BilinearForm;
  class LinearForm;
  class Mesh;
  class BoundaryCondition;
  class Function;
  class TimeDependentODE;
  class TimeStepper;

  /// This class implements the solution functionality for time dependent PDEs.

  class TimeDependentPDE : public GenericPDE
  {
  public:

    /// Define a time dependent PDE with boundary conditions
    TimeDependentPDE(BilinearForm& a, LinearForm& L, Mesh& mesh,
		     BoundaryCondition& bc, int N, real k, real T);

    /// Destructor
    ~TimeDependentPDE();

    /// Solve PDE (in general a mixed system)
    virtual uint solve(Function& u);

    /// Compute right hand side dotu = f(u)
    virtual void fu(const Vector& x, Vector& dotx, real t);

    virtual void init(Function& U);

    virtual void save(Function& U, real t);

    virtual void preparestep();
    virtual void prepareiteration();

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

    Vector* x;
    Vector* dotx;
    real k;

  protected:

    BilinearForm* _a;
    LinearForm* _Lf;
    Mesh* _mesh;
    BoundaryCondition* _bc;

    TimeDependentODE* ode;
    TimeStepper* ts;

    int N;
    real t;
    real T;

  };

  class TimeDependentODE : public ODE
  {
  public:
    TimeDependentODE(TimeDependentPDE& pde, int N, real T);
    real u0(unsigned int i);
    virtual real timestep(real t, real k0) const;
    // Evaluate right-hand side (mono-adaptive version)
    using ODE::f;
    virtual void f(const real u[], real t, real y[]);
    virtual bool update(const real u[], real t, bool end);
    
  protected:
    TimeDependentPDE* pde;
  };


}

#endif

#endif
