// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEW_FEM_H
#define __NEW_FEM_H

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
    
    static void assembleInterior(NewPDE& pde, Mesh& mesh, Matrix& A);
    static void assembleBoundary(NewPDE& pde, Mesh& mesh, Matrix& A);
    static void assembleBoundaryTri(NewPDE& pde, Mesh& mesh, Matrix& A);
    static void assembleBoundaryTet(NewPDE& pde, Mesh& mesh, Matrix& A);

    static void assembleInterior(NewPDE& pde, Mesh& mesh, Vector& b);
    static void assembleBoundary(NewPDE& pde, Mesh& mesh, Vector& b);
    static void assembleBoundaryTri(NewPDE& pde, Mesh& mesh, Vector& b);
    static void assembleBoundaryTet(NewPDE& pde, Mesh& mesh, Vector& b);

    /// Allocate matrix
    void alloc(Matrix &A, Mesh &mesh, NewPDE& pde);
      
    /// Allocate vector
    void alloc(Vector &b, Mesh &mesh, NewPDE& pde);
  
    /// FIXME: Temporary strong implementation of BC
    void setBC(Mesh& mesh, Matrix& A, NewPDE& pde);
    void setBC(Mesh& mesh, Vector& b, NewPDE& pde);

  };

}

#endif
