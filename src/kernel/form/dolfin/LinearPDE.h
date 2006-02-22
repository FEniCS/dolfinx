// Copyright (C) 2004-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2006
//
// First added:  2004
// Last changed: 2006-02-22

#ifndef __LINEAR_PDE_H
#define __LINEAR_PDE_H

#include <dolfin/GenericPDE.h>
#include <dolfin/Parametrized.h>

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
    Function solve();

    /// Solve PDE (in general a mixed system)
    void solve(Function& u);

    /// Solve PDE for sub functions of mixed system
    void solve(Function& u0, Function& u1);

    /// Solve PDE for sub functions of mixed system
    void solve(Function& u0, Function& u1, Function& u2);

    /// Solve PDE for sub functions of mixed system
    void solve(Function& u0, Function& u1, Function& u2, Function& u3);
    
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

  private:

    //FIXME: Add functions to assemble RHS vector and stiffness matrix
//    // Form RHS vector and Jacobian matrix
//    void form(Matrix& A, Vector& b)

  };

}

#endif
