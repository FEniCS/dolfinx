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
  /// formulation. It is currently assumed that each degree of freedom
  /// is associated with a node. The ID of the node is used as an index
  /// for the degrees of freedom.

  class NewFEM
  {
  public:
    
    /// Assemble linear system
    static void assemble(NewPDE& pde, Mesh& mesh, Matrix& A, Vector& b);
    
    /// Assemble matrix
    static void assemble(NewPDE& pde, Mesh& mesh, Matrix& A);
    
    /// Assemble vector 
    static void assemble(NewPDE& pde, Mesh& mesh, Vector& b);

  };

}

#endif
