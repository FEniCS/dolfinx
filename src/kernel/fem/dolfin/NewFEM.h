// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEW_FEM_H
#define __NEW_FEM_H

#include <dolfin/constants.h>

namespace dolfin
{

  class NewPDE;
  class BilinearForm;
  class LinearForm;
  class Mesh;
  class Matrix;
  class NewMatrix;
  class NewVector;
  class Vector;

  /// Automated assembly of a linear system from a given partial differential
  /// equation, specified as a variational problem: Find u in V such that
  ///
  ///     a(u,v) = L(v) for all v in V,
  ///
  /// where a(.,.) is a given bilinear form and L(.) is a given linear form.

  class NewFEM
  {
  public:
    
    /// Assemble linear system
    static void assemble(NewPDE& pde, Mesh& mesh, Matrix& A, Vector& b);
    
    /// Assemble matrix for bilinear form
    static void assemble(BilinearForm& a, Mesh& mesh, Matrix& A);
    
    /// Assemble vector for linear form
    static void assemble(LinearForm& L, Mesh& mesh, Vector& b);

    /// Testing PETSc
    static void testPETSc(BilinearForm& a, Mesh& mesh, NewMatrix& A);
    static void testPETSc(LinearForm& L, Mesh& mesh, NewVector& b);

  private:
    
    // Assemble matrix
    static void assembleInterior    (BilinearForm& a, Mesh& mesh, Matrix& A);
    static void assembleBoundary    (BilinearForm& a, Mesh& mesh, Matrix& A);
    static void assembleBoundaryTri (BilinearForm& a, Mesh& mesh, Matrix& A);
    static void assembleBoundaryTet (BilinearForm& a, Mesh& mesh, Matrix& A);

    // Assemble vector
    static void assembleInterior    (LinearForm& L, Mesh& mesh, Vector& b);
    static void assembleBoundary    (LinearForm& L, Mesh& mesh, Vector& b);
    static void assembleBoundaryTri (LinearForm& L, Mesh& mesh, Vector& b);
    static void assembleBoundaryTet (LinearForm& L, Mesh& mesh, Vector& b);

    // Allocate global matrix
    static void alloc(Matrix& A, const NewFiniteElement& element, Mesh& mesh);

    // Allocate global vector
    static void alloc(Vector& b, const NewFiniteElement& element, Mesh& mesh);
  
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
