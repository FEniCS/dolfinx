// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

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

  /// Automated assembly of a linear system from a given partial differential
  /// equation, specified as a variational problem: Find u in V such that
  ///
  ///     a(u,v) = L(v) for all v in V,
  ///
  /// where a(.,.) is a given bilinear form and L(.) is a given linear form.

  class NewFEM
  {
  public:

    static void assemble(BilinearForm& a, LinearForm& L, Mesh& mesh, 
			 NewMatrix& A, NewVector& b); 
    static void assemble(BilinearForm& a, Mesh& mesh, NewMatrix& A);
    static void assemble(LinearForm& L, Mesh& mesh, NewVector& b);

  private:

    // Allocate element matrix (use arrays to improve speed)
    static real** allocElementMatrix(const NewFiniteElement& element);
    
    // Allocate element vector (use arrays to improve speed)
    static real* allocElementVector(const NewFiniteElement& element);

    // Delete element matrix
    static void freeElementMatrix(real**& AK, const NewFiniteElement& element);

    // Delete element vector
    static void freeElementVector(real*& bK, const NewFiniteElement& element);
    
    // Count the degrees of freedom
    static uint size(Mesh& mesh, const NewFiniteElement& element);

  };

}

#endif
