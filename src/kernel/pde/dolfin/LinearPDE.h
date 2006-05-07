// Copyright (C) 2004-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2006.
//
// First added:  2004
// Last changed: 2006-05-07

#ifndef __LINEAR_PDE_H
#define __LINEAR_PDE_H

#ifdef HAVE_PETSC_H

#include <dolfin/GenericPDE.h>

namespace dolfin
{

  class BilinearForm;
  class LinearForm;
  class Mesh;
  class BoundaryCondition;
  class Function;

  /// This class implements the solution functionality for linear PDEs.

  class LinearPDE : public GenericPDE
  {
  public:

    /// Define a static linear PDE with natural boundary conditions
    LinearPDE(BilinearForm& a, LinearForm& L, Mesh& mesh);

    /// Define a static linear PDE with Dirichlet boundary conditions
    LinearPDE(BilinearForm& a, LinearForm& L, Mesh& mesh, BoundaryCondition& bc);

    /// Destructor
    ~LinearPDE();

    /// Solve PDE (in general a mixed system)
    uint solve(Function& u);

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
    // FIXME: _L for the LinearForm is consistent, but breaks under Cygwin (as does _C, _X, etc)
    LinearForm* _Lf;
    Mesh* _mesh;
    BoundaryCondition* _bc;

  };

}

#endif

#endif
