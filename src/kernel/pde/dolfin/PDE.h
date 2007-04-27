// Copyright (C) 2004-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2006.
//
// First added:  2004
// Last changed: 2007-04-26

#ifndef __PDE_H
#define __PDE_H

#include <dolfin/Array.h>
#include <dolfin/Parametrized.h>

namespace dolfin
{

  class Form;
  class Mesh;
  class BoundaryCondition;
  class Function;
  class GenericPDE;

  /// A PDE represents a partial differential equation in variational
  /// form. The PDE can be either linear or nonlinear.
  ///
  /// For a linear PDE, the variational problem reads as follows:
  /// Find u in V such that
  ///
  ///     a(v, u) = L(v) for all v in V',
  ///
  /// where a is a bilinear form and L is a linear form.
  ///
  /// For a nonlinear PDE, the variational problem reads as follows:
  /// Find u in V such that
  ///
  ///     L_u(v) = 0 for all v in V',
  ///
  /// where L_u is a linear form for each fixed u. The derivative
  /// (linearization) of L_u with respect to u must also be supplied
  /// in the form of a bilinear form a = a(v, u) = L'_u(v, u).

  class PDE : public Parametrized
  {
  public:

    /// PDE types
    enum Type {linear, nonlinear};

    /// Define a PDE with natural boundary conditions
    PDE(Form& a, Form& L, Mesh& mesh, Type type = linear);

    /// Define a PDE with a single Dirichlet boundary condition
    PDE(Form& a, Form& L, Mesh& mesh, BoundaryCondition& bc, Type type = linear);
    
    /// Define a PDE with a set of Dirichlet boundary conditions
    PDE(Form& a, Form& L, Mesh& mesh, Array<BoundaryCondition*>& bcs, Type type = linear);

    /// Destructor
    ~PDE();

    /// Solve PDE
    void solve(Function& u);

    /// Return type of PDE
    Type type() const;

  private:
    
    // Initialize
    void init(Form& a, Form& L, Mesh& mesh, Array<BoundaryCondition*>& bcs);

    // Pointer to current implementation (letter base class)
    GenericPDE* pde;

    // PDE type
    Type _type;

  };

}

#endif
