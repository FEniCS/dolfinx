// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Fredrik Bengzon and Johan Jansson, 2004.

#ifndef __FEM_H
#define __FEM_H

#include <dolfin/FiniteElement.h>


namespace dolfin {

  class PDE;
  class Mesh;
  class Matrix;
  class Vector;
  class Map;
  class Quadrature;
  class FiniteElementMethod;
  
  /// Automated assembly of a linear system from a given variational
  /// formulation. It is currently assumed that each degree of freedom
  /// is associated with a node. The ID of the node is used as an index
  /// for the degrees of freedom.

  class FEM {
  public:
    
    /// Assemble linear system
    static void assemble(PDE& pde, Mesh& mesh, Matrix& A, Vector& b);

    /// Assemble matrix
    static void assemble(PDE& pde, Mesh& mesh, Matrix& A);

    /// Assemble vector 
    static void assemble(PDE& pde, Mesh& mesh, Vector& b);
    
    /// Assemble linear system for user-defined method
    static void assemble(PDE& pde, Mesh& mesh, Matrix& A, Vector& b,
			 FiniteElementMethod& method);

    /// Assemble matrix for user-defined method
    static void assemble(PDE& pde, Mesh& mesh, Matrix& A,
			 FiniteElementMethod& method);

    /// Assemble vector for user-defined method
    static void assemble(PDE& pde, Mesh& mesh, Vector& b,
			 FiniteElementMethod& method);
    
  private:

    // Assemble matrix
    static void assemble(PDE& pde, Mesh& mesh, Matrix& A,
			 FiniteElement::Vector& element, Map& map,
			 Quadrature& quadrature);
    
    // Assemble vector
    static void assemble(PDE& pde, Mesh& mesh, Vector& b,
			 FiniteElement::Vector& element, Map& map,
			 Quadrature& quadrature);

    // Set boundary conditions for matrix
    static void setBC(Mesh& mesh, Matrix& A, FiniteElement::Vector& element);

    // Set boundary conditions for vector
    static void setBC(Mesh& mesh, Vector& b, FiniteElement::Vector& element);

    // Initialize matrix
    static void alloc(Matrix& A, Mesh& mesh, FiniteElement::Vector& element);

    // Initialize vector
    static void alloc(Vector& b, Mesh& mesh, FiniteElement::Vector& element);

    // Default method
    static FiniteElementMethod method;

  };

}

#endif
