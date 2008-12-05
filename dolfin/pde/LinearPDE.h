// Copyright (C) 2004-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells 2006-2008.
// Modified by Dag Lindbo, 2008.
//
// First added:  2004
// Last changed: 2008-12-05

#ifndef __LINEAR_PDE_H
#define __LINEAR_PDE_H

#include <vector>

#include <dolfin/parameter/Parametrized.h>
#include <dolfin/la/enums_la.h>

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
    LinearPDE(const Form& a,
              const Form& L, 
              dolfin::MatrixType matrix_type=nonsymmetric);
    
    /// Define a linear PDE with a single Dirichlet boundary condition
    LinearPDE(const Form& a,
              const Form& L,
              const BoundaryCondition& bc, 
              dolfin::MatrixType matrix_type=nonsymmetric);

    // FIXME: Pointers need to be const here to work with SWIG. Is there a fix for this?
    
    /// Define a linear PDE with a set of Dirichlet boundary conditions
    LinearPDE(const Form& a,
              const Form& L,
              std::vector<BoundaryCondition*>& bcs, 
              dolfin::MatrixType matrix_type=nonsymmetric);

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
    const Form& a;
    
    // The linear form
    const Form& L;

    // The boundary conditions
    Array<const BoundaryCondition*> bcs;
  
    // Symmetry of the bilinear form
    dolfin::MatrixType matrix_type;

  };
}

#endif
