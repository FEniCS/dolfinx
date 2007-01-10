// Copyright (C) 2006 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2006.
//
// First added:  2005
// Last changed: 2006-08-21

#ifndef __TIME_DEPENDENT_PDE_H
#define __TIME_DEPENDENT_PDE_H

//#ifdef HAVE_PETSC_H

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
		     BoundaryCondition& bc, int N, real T);

    /// Destructor
    ~TimeDependentPDE();

    /// Solve PDE (in general a mixed system)
    virtual uint solve(Function& u);

    /// Compute initial value
    virtual void u0(uBlasVector& u);

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
    void u0(uBlasVector& u);

    // Evaluate right-hand side (mono-adaptive version)
    using ODE::f;
    virtual void f(const uBlasVector& u, real t, uBlasVector& y);
    virtual bool update(const uBlasVector& u, real t, bool end);
    
  protected:
    TimeDependentPDE* pde;
  };


}

//#endif

#endif
