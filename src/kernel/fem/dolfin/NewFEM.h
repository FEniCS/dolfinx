// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEW_FEM_H
#define __NEW_FEM_H

#include <dolfin/NewArray.h>

namespace dolfin
{

  class NewPDE;
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
    
    /// Assemble matrix
    static void assemble(NewPDE& pde, Mesh& mesh, Matrix& A);
    
    /// Assemble vector 
    static void assemble(NewPDE& pde, Mesh& mesh, Vector& b);

  private:
    
    /// Assemble matrix
    static void assembleInterior    (NewPDE& pde, Mesh& mesh, Matrix& A);
    static void assembleBoundary    (NewPDE& pde, Mesh& mesh, Matrix& A);
    static void assembleBoundaryTri (NewPDE& pde, Mesh& mesh, Matrix& A);
    static void assembleBoundaryTet (NewPDE& pde, Mesh& mesh, Matrix& A);

    /// Assemble vector
    static void assembleInterior    (NewPDE& pde, Mesh& mesh, Vector& b);
    static void assembleBoundary    (NewPDE& pde, Mesh& mesh, Vector& b);
    static void assembleBoundaryTri (NewPDE& pde, Mesh& mesh, Vector& b);
    static void assembleBoundaryTet (NewPDE& pde, Mesh& mesh, Vector& b);

    /// Allocate global matrix
    static void alloc(const NewPDE& pde, Mesh& mesh, Matrix &A);

    /// Allocate global vector
    static void alloc(const NewPDE& pde, Mesh& mesh, Vector& b);
  
    /// Allocate element matrix
    static void alloc(const NewPDE& pde, real**& A);

    /// Allocate element vector
    static void alloc(const NewPDE& pde, NewArray<real>& b);

    /// FIXME: Temporary strong implementation of BC
    static void setBC(const NewPDE& pde, Mesh& mesh, Matrix& A);
    static void setBC(const NewPDE& pde, Mesh& mesh, Vector& b);

  };

}

#endif
