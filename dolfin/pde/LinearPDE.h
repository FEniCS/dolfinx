// Copyright (C) 2004-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2006, 2007.
// Modified by Dag Lindbo, 2008.
//
// First added:  2004
// Last changed: 2007-12-09

#ifndef __LINEAR_PDE_H
#define __LINEAR_PDE_H

#include <dolfin/common/Array.h>
#include <dolfin/parameter/Parametrized.h>
#include <dolfin/la/Vector.h>
#include <dolfin/fem/DofMapSet.h>

namespace dolfin
{

  class Form;
  class Mesh;
  class BoundaryCondition;
  class Function;

  /// A LinearPDE represents a (system of) linear partial differential
  /// equation(s) in variational form: Find u in V such that
  ///
  ///     a(v, u) = L(v) for all v in V',
  ///
  /// where a is a bilinear form and L is a linear form.

  class LinearPDE : public Parametrized
  {
  public:

    /// Define a linear PDE with natural boundary conditions
    LinearPDE(Form& a, Form& L, Mesh& mesh);
    
    /// Define a linear PDE with a single Dirichlet boundary condition
    LinearPDE(Form& a, Form& L, Mesh& mesh, BoundaryCondition& bc);
    
    /// Define a linear PDE with a set of Dirichlet boundary conditions
    LinearPDE(Form& a, Form& L, Mesh& mesh, Array<BoundaryCondition*>& bcs);

    /// Destructor
    ~LinearPDE();
    
    /// Solve PDE system
    void solve(Function& u);

    /// Solve PDE system and extract sub functions
    void solve(Function& u0, Function& u1);

    /// Solve PDE system and extract sub functions
    void solve(Function& u0, Function& u1, Function& u2);

  private:
    
    // The bilinear form
    Form& a;
    
    // The linear form
    Form& L;

    // The mesh
    Mesh& mesh;

    // The boundary conditions
    Array<BoundaryCondition*> bcs;
  };
}

#endif
