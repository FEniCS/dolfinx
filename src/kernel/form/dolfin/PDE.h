// Copyright (C) 2004-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004
// Last changed: 2006-02-10

#ifndef __PDE_H
#define __PDE_H

#include <dolfin/Parametrized.h>

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

    /// Define a static linear PDE with natural boundary conditions
    PDE(BilinearForm& a, LinearForm& L, Mesh& mesh);

    /// Define a static linear PDE with Dirichlet boundary conditions
    PDE(BilinearForm& a, LinearForm& L, Mesh& mesh, BoundaryCondition& bc);

    /// Destructor
    ~PDE();

    /// Solve PDE
    Function solve();

    /// Solve PDE
    void solve(Function& u);
    
    /// Return the bilinear form a(.,.)
    BilinearForm& a();

    /// Return the linear form L(.,.)
    LinearForm& L();

    /// Return the mesh
    Mesh& mesh();

  protected:

    BilinearForm* _a;
    LinearForm* _L;
    Mesh* _mesh;
    BoundaryCondition* _bc;

  };

}

#endif
