// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#ifndef __NEW_FEM_H
#define __NEW_FEM_H

#include <dolfin/constants.h>

namespace dolfin
{

  class BilinearForm;
  class LinearForm;
  class Mesh;
  class NewMatrix;
  class NewVector;
  class NewFiniteElement;
  class NewBoundaryCondition;

  /// Automated assembly of a linear system from a given partial differential
  /// equation, specified as a variational problem: Find u in V such that
  ///
  ///     a(u,v) = L(v) for all v in V,
  ///
  /// where a(.,.) is a given bilinear form and L(.) is a given linear form.

  class NewFEM
  {
  public:

    /// Assemble bilinear form
    static void assemble(BilinearForm& a, NewMatrix& A, Mesh& mesh,
			 const NewFiniteElement& element);

    /// Assemble linear form
    static void assemble(LinearForm& L, NewVector& b, Mesh& mesh,
			 const NewFiniteElement& element);

    /// Assemble bilinear and linear forms
    static void assemble(BilinearForm& a, LinearForm& L, 
			 NewMatrix& A, NewVector& b, Mesh& mesh,
			 const NewFiniteElement& element);

    /// Assemble bilinear and linear forms (including Dirichlet boundary conditions)
    static void assemble(BilinearForm& a, LinearForm& L, 
			 NewMatrix& A, NewVector& b, Mesh& mesh,
			 const NewFiniteElement& element, NewBoundaryCondition& bc);

    /// Set Dirichlet boundary conditions
    static void setBC(NewMatrix& A, NewVector& b, Mesh& mesh,
		      NewBoundaryCondition& bc);

  private:

    // Count the degrees of freedom
    static uint size(Mesh& mesh, const NewFiniteElement& element);

  };

}

#endif
