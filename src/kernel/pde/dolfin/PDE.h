// Copyright (C) 2004-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2006.
//
// First added:  2004
// Last changed: 2006-02-22

#ifndef __PDE_H
#define __PDE_H

#include <dolfin/Parametrized.h>
#include <dolfin/GenericPDE.h>

namespace dolfin
{

  class BilinearForm;
  class LinearForm;
  class Mesh;
  class BoundaryCondition;
  class Function;

  /// A PDE represents a (linearized) partial differential equation,
  /// given by a variation problem of the form: Find u in V such that
  ///
  ///     a(v, u) = L(v) for all v in V,
  ///
  /// where a(.,.) is a given bilinear form and L(.) is a given linear form.

  class PDE : public Parametrized
  {
  public:

    /// PDE types
    enum Type { linear, nonlinear };

    /// Define a static linear PDE with natural boundary conditions
    PDE(BilinearForm& a, LinearForm& L, Mesh& mesh);

    /// Define a static linear PDE with Dirichlet boundary conditions
    PDE(BilinearForm& a, LinearForm& L, Mesh& mesh, BoundaryCondition& bc);

    /// Define a PDE with Natural boundary conditions
    PDE(BilinearForm& a, LinearForm& L, Mesh& mesh, Type pde_type);

    /// Define a PDE with Dirichlet boundary conditions
    PDE(BilinearForm& a, LinearForm& L, Mesh& mesh, BoundaryCondition& bc, Type pde_type);

    /// Destructor
    ~PDE();

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

    /// Return type of PDE
    inline Type type() const;

  private:

    // Pointer to current implementation (letter base class)
    GenericPDE* pde;

    // PDE type (linear, nonlinear, . . .)
    Type _type;

  };

}

#endif
