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

  class FEM
  {
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
			 Quadrature& interior_quadrature, 
			 Quadrature& boundary_quadrature);
			     
    /// Assemble matrix for interior
    static void assembleInterior(PDE& pde, Mesh& mesh, Matrix& A, 
				 FiniteElement::Vector& element, Map& map,
				 Quadrature& interior_quadrature, 
				 Quadrature& boundary_quadrature);
				
    /// Assemble matrix for boundary 
    static void assembleBoundary(PDE& pde, Mesh& mesh, Matrix& A,
				 FiniteElement::Vector& element, Map& map,
				 Quadrature& interior_quadrature, 
				 Quadrature& boundary_quadrature);

    /// Assemble matrix for boundary (triangular mesh)
    static void assembleBoundaryTri(PDE& pde, Mesh& mesh, Matrix& A,
				    FiniteElement::Vector& element, Map& map,
				    Quadrature& interior_quadrature, 
				    Quadrature& boundary_quadrature);

    /// Assemble matrix for boundary (tetrahedral mesh)
    static void assembleBoundaryTet(PDE& pde, Mesh& mesh, Matrix& A,
				    FiniteElement::Vector& element, Map& map,
				    Quadrature& interior_quadrature, 
				    Quadrature& boundary_quadrature);

    // Assemble vector
    static void assemble(PDE& pde, Mesh& mesh, Vector& b,
			 FiniteElement::Vector& element, Map& map,
			 Quadrature& interior_quadrature, 
			 Quadrature& boundary_quadrature);

    /// Assemble vector for interior
    static void assembleInterior(PDE& pde, Mesh& mesh, Vector& b,
				 FiniteElement::Vector& element, Map& map,
				 Quadrature& interior_quadrature, 
				 Quadrature& boundary_quadrature);
      
    /// Assemble vector for boundary 
    static void assembleBoundary(PDE& pde, Mesh& mesh, Vector& b,
				 FiniteElement::Vector& element, Map& map,
				 Quadrature& interior_quadrature, 
				 Quadrature& boundary_quadrature);

    /// Assemble vector for boundary (triangular mesh)
    static void assembleBoundaryTri(PDE& pde, Mesh& mesh, Vector& b,
				    FiniteElement::Vector& element, Map& map,
				    Quadrature& interior_quadrature, 
				    Quadrature& boundary_quadrature);

    /// Assemble vector for boundary (tetrahedral mesh)
    static void assembleBoundaryTet(PDE& pde, Mesh& mesh, Vector& b,
				    FiniteElement::Vector& element, Map& map,
				    Quadrature& interior_quadrature, 
				    Quadrature& boundary_quadrature);
    
    // Set boundary conditions for matrix
    static void setBC(Mesh& mesh, Matrix& A, FiniteElement::Vector& element);

    // Set boundary conditions for vector
    static void setBC(Mesh& mesh, Vector& b, FiniteElement::Vector& element);

    // Initialize matrix
    static void alloc(Matrix& A, Mesh& mesh, FiniteElement::Vector& element);

    // Initialize vector
    static void alloc(Vector& b, Mesh& mesh, FiniteElement::Vector& element);

  };

}

#endif
