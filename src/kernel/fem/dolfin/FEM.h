// Copyright (C) 2004-2006 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2006.
// Modified by Kristian Oelgaard 2006.
//
// First added:  2004-05-19
// Last changed: 2006-09-29

#ifndef __FEM_H
#define __FEM_H

#include <dolfin/constants.h>
#include <dolfin/AffineMap.h>

#include <dolfin/GenericMatrix.h>
#include <dolfin/GenericVector.h>
#include <dolfin/DenseMatrix.h>
#include <dolfin/Vector.h>
#include <dolfin/SparseMatrix.h>
#include <dolfin/Vector.h>

#include <dolfin/Mesh.h>
#include <dolfin/BoundaryMesh.h>
#include <dolfin/BoundaryValue.h>
#include <dolfin/BoundaryCondition.h>

#include <dolfin/FiniteElement.h>
#include <dolfin/BilinearForm.h>
#include <dolfin/LinearForm.h>
#include <dolfin/Functional.h>

// FIXME: Ensure constness where appropriate

namespace dolfin
{

  /// Automated assembly of a linear system from a given partial differential
  /// equation, specified as a variational problem: Find U in V such that
  ///
  ///     a(v, U) = L(v) for all v in V,
  ///
  /// where a(.,.) is a given bilinear form and L(.) is a given linear form.

  class FEM
  {
  public:

    /// Assemble bilinear and linear forms
    static void assemble(BilinearForm& a, LinearForm& L,
			 GenericMatrix& A, GenericVector& b,
			 Mesh& mesh);

    /// Assemble bilinear and linear forms including boundary conditions
    static void assemble(BilinearForm& a, LinearForm& L,
			 GenericMatrix& A, GenericVector& b,
			 Mesh& mesh, BoundaryCondition& bc);

    /// Assemble bilinear form
    static void assemble(BilinearForm& a, GenericMatrix& A, Mesh& mesh);
    
    /// Assemble linear form
    static void assemble(LinearForm& L, GenericVector& b, Mesh& mesh);

    /// Assemble functional
    static real assemble(Functional& M, Mesh& mesh);
   
    /// Apply boundary conditions to matrix and vector
    static void applyBC(GenericMatrix& A, GenericVector& b, Mesh& mesh,
			FiniteElement& element, BoundaryCondition& bc);
    
    /// Apply boundary conditions to matrix
    static void applyBC(GenericMatrix& A, Mesh& mesh,
			FiniteElement& element, BoundaryCondition& bc);

    /// Apply boundary conditions to vector
    static void applyBC(GenericVector& b, Mesh& mesh,
			FiniteElement& element, BoundaryCondition& bc);

    /// Assemble boundary conditions into residual vector, with b = x - bc
    /// for Dirichlet and b = x - bc for Neumann boundary conditions
    static void assembleResidualBC(GenericMatrix& A, GenericVector& b,
				   const GenericVector& x, Mesh& mesh,
				   FiniteElement& element, BoundaryCondition& bc);
    
    /// Assemble boundary conditions into residual vector, with b = x - bc
    /// for Dirichlet and b = x - bc for Neumann boundary conditions
    static void assembleResidualBC(GenericVector& b,
				   const GenericVector& x, Mesh& mesh,
				   FiniteElement& element, BoundaryCondition& bc);

    /// Lump matrix (cannot mix matrix vector and matrix types when lumping)
    template < class A, class X > 
    static void lump(const A& M, X& m) { M.lump(m); }

    /// Count the degrees of freedom
    static uint size(Mesh& mesh, const FiniteElement& element);

    /// Display assembly data (useful for debugging)
    static void disp(Mesh& mesh, const FiniteElement& element);
      
  private:

    /// Common assembly for bilinear and linear forms
    static void assembleCommonOld(BilinearForm* a, LinearForm* L, Functional* M,
			       GenericMatrix* A, GenericVector* b, real* val, Mesh& mesh);

    /// Create iterator and call function to apply boundary conditions
    static void applyCommonBCOld(GenericMatrix* A, GenericVector* b, const GenericVector* x,
			      Mesh& mesh, FiniteElement& element, BoundaryCondition& bc);

    /// Check that dimension of the mesh matches the form
    static void checkDimensions(const BilinearForm& a, const Mesh& mesh);

    /// Check that dimension of the mesh matches the form
    static void checkDimensions(const LinearForm& L, const Mesh& mesh);

    /// Estimate the maximum number of nonzeros in each row
    static uint estimateNonZeros(Mesh& mesh, const FiniteElement& element);

    /// Check actual number of nonzeros in each row
    static void countNonZeros(const GenericMatrix& A, uint nz);

    // Since the current mesh interface is dimension-dependent, the functions
    // assembleCommon() and applyCommonBC() need to be templated. They won't
    // have to be when the new mesh interface is in place.

    static void assembleCommon(BilinearForm* a, LinearForm* L, Functional* M,
			       GenericMatrix* A, GenericVector* b, real* val,
			       Mesh& mesh);

    static void applyCommonBC(GenericMatrix* A, GenericVector* b, 
			      const GenericVector* x, Mesh& mesh,
			      FiniteElement& element, BoundaryCondition& bc);

    /// Assemble bilinear form for an element
    static void assembleElement(BilinearForm& a, GenericMatrix& A, 
      const Mesh& mesh, const Cell& cell, AffineMap& map, const int facetID);

    /// Assemble linear form for an element
    static void assembleElement(LinearForm& L, GenericVector& b, const Mesh& mesh, 
                      const Cell& cell, AffineMap& map, const int facetID);

    /// Assemble fucntional for an element
    static void assembleElement(Functional& M, real& val, AffineMap& map,
                                   const int facetID);
  };
  //-------------------------------------------------------------------------------
  // Template and inline function definitions  
  //-------------------------------------------------------------------------------
  inline void FEM::assembleElement(BilinearForm& a, GenericMatrix& A, 
      const Mesh& mesh, const Cell& cell, AffineMap& map, const int facetID)
  {
    // Update form
    a.update(map);
            
    // Compute maps from local to global degrees of freedom
    a.test().nodemap(a.test_nodes, cell, mesh);
    a.trial().nodemap(a.trial_nodes, cell, mesh);
            
    // Compute element matrix 
    if( facetID < 0 )
      a.eval(a.block, map);
    else
      a.eval(a.block, map, facetID);

    // Add element matrix to global matrix
    A.add(a.block, a.test_nodes, a.test().spacedim(), a.trial_nodes, a.trial().spacedim());
  }
  //-----------------------------------------------------------------------------
  inline void FEM::assembleElement(LinearForm& L, GenericVector& b, 
    const Mesh& mesh, const Cell& cell, AffineMap& map, const int facetID)
  {
    // Update form
    L.update(map);
            
    // Compute map from local to global degrees of freedom
    L.test().nodemap(L.test_nodes, cell, mesh);
            
    // Compute element vector 
    if( facetID < 0 )
      L.eval(L.block, map);
    else
      L.eval(L.block, map, facetID);

    // Add element vector to global vector
    b.add(L.block, L.test_nodes, L.test().spacedim());
  }
  //-----------------------------------------------------------------------------
  inline void FEM::assembleElement(Functional& M, real& val, AffineMap& map,
                                        const int facetID)
  {
    // Update form
    M.update(map);
            
    // Compute element entry
    if( facetID < 0 )
      M.eval(M.block, map);
    else
      M.eval(M.block, map, facetID);
    
    // Add element entry to global value
    val += M.block[0];
  }
  //-----------------------------------------------------------------------------

}

#endif
