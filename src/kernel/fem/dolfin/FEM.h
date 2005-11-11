// Copyright (C) 2004-2005 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells
//
// First added:  2004-05-19
// Last changed: 2005-11-11

#ifndef __FEM_H
#define __FEM_H

#include <dolfin/constants.h>

namespace dolfin
{
  class BilinearForm;
  class LinearForm;
  class Mesh;
  class Matrix;
  class Vector;
  class FiniteElement;
  class BoundaryCondition;

  /// Automated assembly of a linear system from a given partial differential
  /// equation, specified as a variational problem: Find u in V such that
  ///
  ///     a(v, u) = L(v) for all v in V,
  ///
  /// where a(.,.) is a given bilinear form and L(.) is a given linear form.

  class FEM
  {
  public:

    /// Assemble bilinear form
    static void assemble(BilinearForm& a, Matrix& A, Mesh& mesh);

    /// Assemble linear form
    static void assemble(LinearForm& L, Vector& b, Mesh& mesh);

    /// Assemble bilinear and linear forms
    static void assemble(BilinearForm& a, LinearForm& L, 
			 Matrix& A, Vector& b, Mesh& mesh);
    
    /// Assemble bilinear and linear forms (including Dirichlet boundary conditions)
    static void assemble(BilinearForm& a, LinearForm& L, 
			 Matrix& A, Vector& b, Mesh& mesh,
			 BoundaryCondition& bc);
    
    /// Apply boundary conditions to matrix and vector 
    static void applyBC(Matrix& A, Vector& b, Mesh& mesh,
			const FiniteElement& element, BoundaryCondition& bc);
    
    /// Apply boundary conditions to matrix 
    static void applyBC(Matrix& A, Mesh& mesh, const FiniteElement& element, 
			BoundaryCondition& bc);

    /// Apply boundary conditions to vector 
    static void applyBC(Vector& b, Mesh& mesh, const FiniteElement& element,
			BoundaryCondition& bc);

    /// Lump matrix
    static void lump(const Matrix& M, Vector& m);

    /// Count the degrees of freedom
    static uint size(const Mesh& mesh, const FiniteElement& element);

    /// Estimate the maximum number of nonzeros in each row
    static uint nzsize(const Mesh& mesh, const FiniteElement& element);

    /// Display assembly data (useful for debugging)
    static void disp(const Mesh& mesh, const FiniteElement& element);
      
  private:

    /// Apply boundary conditions on triangular mesh
    static void applyBC_2D(Matrix& A, Vector& b, Mesh& mesh,
			   const FiniteElement& element, BoundaryCondition& bc);

    /// Apply boundary conditions to matrix on triangular mesh
    static void applyBC_2D(Matrix& A, Mesh& mesh, const FiniteElement& element,
         BoundaryCondition& bc);

    /// Apply boundary conditions to vector on triangular mesh
    static void applyBC_2D(Vector& b, Mesh& mesh, const FiniteElement& element,
        BoundaryCondition& bc);

    /// Apply boundary conditions on tetrahedral mesh
    static void applyBC_3D(Matrix& A, Vector& b, Mesh& mesh,
			   const FiniteElement& element, BoundaryCondition& bc);

    /// Apply boundary conditions to matrix on tetrahedral mesh
    static void applyBC_3D(Matrix& A, Mesh& mesh, const FiniteElement& element,
         BoundaryCondition& bc);

    /// Apply boundary conditions to vector on tetrahedral mesh
    static void applyBC_3D(Vector& b, Mesh& mesh, const FiniteElement& element,
         BoundaryCondition& bc);

    /// Check that dimension of the mesh matches the form
    static void checkdims(const BilinearForm& a, const Mesh& mesh);

    /// Check that dimension of the mesh matches the form
    static void checkdims(const LinearForm& L, const Mesh& mesh);

    /// Check number of nonzeros in each row
    static void checknz(const Matrix& A, uint nz);

  };

}

#endif
