// Copyright (C) 2004-2006 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2006.
// Modified by Kristian Oelgaard 2006.
//
// First added:  2004-05-19
// Last changed: 2006-12-05

#ifndef __FEM_H
#define __FEM_H

#include <dolfin/constants.h>

// FIXME: Ensure constness where appropriate

namespace dolfin
{
  
  class BilinearForm;
  class LinearForm;
  class Functional;
  class Mesh;
  class Cell;
  class Point;
  class GenericMatrix;
  class GenericVector;
  class FiniteElement;
  class BoundaryCondition;
  class AffineMap;
  class DofMap;

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

    /// Apply boundary conditions with b = bc - x at Dirichlet nodes
    static void applyResidualBC(GenericMatrix& A, GenericVector& b,
                                const GenericVector& x, Mesh& mesh,
                                FiniteElement& element, BoundaryCondition& bc);

    /// Apply boundary conditions with b = bc - x at Dirichlet nodes
    static void applyResidualBC(GenericVector& b,
                                const GenericVector& x, Mesh& mesh,
                                FiniteElement& element, BoundaryCondition& bc);

    /// Lump matrix (cannot mix matrix vector and matrix types when lumping)
    template < class A, class X > static void lump(const A& M, X& m) { M.lump(m); }

    /// Count the degrees of freedom. 
    /// FIXME: This function is used by Functions, but the size should be computed
    /// via DofMap 
    static uint size(Mesh& mesh, const FiniteElement& element);

    /// Display assembly data (useful for debugging)
    static void disp(Mesh& mesh, const FiniteElement& element);
      
  private:

    /// Common assembly handles all cases
    static void assembleCommon(BilinearForm* a, LinearForm* L, Functional* M,
			       GenericMatrix* A, GenericVector* b, real* val,
			       Mesh& mesh);

    /// Common application of boundary conditions handles all cases
    static void applyCommonBC(GenericMatrix* A, GenericVector* b, 
			      const GenericVector* x, Mesh& mesh,
			      FiniteElement& element, BoundaryCondition& bc);


    /// Check that dimension of the mesh matches the form
    static void checkDimensions(const BilinearForm& a, const Mesh& mesh);

    /// Check that dimension of the mesh matches the form
    static void checkDimensions(const LinearForm& L, const Mesh& mesh);

    /// Estimate the maximum number of nonzeros in each row
    static uint estimateNonZeros(Mesh& mesh, const FiniteElement& element);

    /// Assemble element tensor for a bilinear form
    static void assembleElementTensor(BilinearForm& a, GenericMatrix& A, 
                                      const Mesh& mesh, const Cell& cell, 
                                      AffineMap& map, const DofMap& dofmap);

    /// Assemble element tensor for a linear form
    static void assembleElementTensor(LinearForm& L, GenericVector& b, 
                                      const Mesh& mesh, const Cell& cell, 
                                      AffineMap& map, const DofMap& dofmap);

    /// Assemble element tensor for a functional
    static void assembleElementTensor(Functional& M, real& val, AffineMap& map);

    /// Assemble exterior facet tensor for a bilinear form
    static void assembleExteriorFacetTensor(BilinearForm& a, GenericMatrix& A, 
                                            const Mesh& mesh, const Cell& cell,
                                            AffineMap& map, uint facet);

    /// Assemble exterior facet tensor for a linear form
    static void assembleExteriorFacetTensor(LinearForm& L, GenericVector& b,
                                            const Mesh& mesh, const Cell& cell,
                                            AffineMap& map, uint facet);

    /// Assemble exterior facet tensor for a functional
    static void assembleExteriorFacetTensor(Functional& M, real& val,
                                            AffineMap& map, uint facet);

    /// Assemble interior facet tensor for a bilinear form
    static void assembleInteriorFacetTensor(BilinearForm& a, GenericMatrix& A, 
                                            const Mesh& mesh,
                                            const Cell& cell0, const Cell& cell1,
                                            AffineMap& map0, AffineMap& map1,
                                            uint facet0, uint facet1, uint alignment);

    /// Assemble interior facet tensor for a linear form
    static void assembleInteriorFacetTensor(LinearForm& L, GenericVector& b,
                                            const Mesh& mesh,
                                            const Cell& cell0, const Cell& cell1,
                                            AffineMap& map0, AffineMap& map1,
                                            uint facet0, uint facet1, uint alignment);

    /// Assemble interior facet tensor for a functional
    static void assembleInteriorFacetTensor(Functional& M, real& val,
                                            AffineMap& map0, AffineMap& map1,
                                            uint facet0, uint facet1, uint alignment);

    /// Initialize mesh connectivity for use in node map
    static void initConnectivity(Mesh& mesh);
    
    /// Check if the point is in the same plane as the given facet
    static bool onFacet(const Point& p, Cell& facet);
    
  };

}

#endif
