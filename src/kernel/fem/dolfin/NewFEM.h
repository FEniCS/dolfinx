// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEW_FEM_H
#define __NEW_FEM_H

#include <dolfin/NewArray.h>

namespace dolfin
{

  class NewPDE;
  class BilinearForm;
  class LinearForm;
  class Mesh;
  class Matrix;
  class Vector;

  /// Automated assembly of a linear system from a given variational
  /// formulation. 

  class NewFEM
  {
  public:
    
    /// Assemble linear system
    static void assemble(NewPDE& pde, Mesh& mesh, Matrix& A, Vector& b);
    
    /// Assemble matrix for bilinear form
    static void assemble(BilinearForm& a, Mesh& mesh, Matrix& A);
    
    /// Assemble vector for linear form
    static void assemble(LinearForm& L, Mesh& mesh, Vector& b);

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
    static void alloc(const NewFiniteElement& element, Mesh& mesh, Matrix &A);

    // Allocate global vector
    static void alloc(const NewFiniteElement& element, Mesh& mesh, Vector& b);
  
    // Allocate element matrix (use arrays to improve speed)
    static real** allocElementMatrix(const NewFiniteElement& element);
    
    // Allocate element vector (use arrays to improve speed)
    static real* allocElementVector(const NewFiniteElement& element);

    // Delete element matrix
    static void freeElementMatrix(real**& AK, const NewFiniteElement& element);

    // Delete element vector
    static void freeElementVector(real*& bK, const NewFiniteElement& element);
        
  };

}

#endif
