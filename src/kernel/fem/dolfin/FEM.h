// Copyright (C) 2004-2005 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2006
// Modified by Kristian Oelgaard, 2006
//
// First added:  2004-05-19
// Last changed: 2006-05-04

#ifndef __FEM_H
#define __FEM_H

#include <dolfin/DenseMatrix.h>
#include <dolfin/DenseVector.h>
#include <dolfin/constants.h>
#include <dolfin/AffineMap.h>
#include <dolfin/Mesh.h>
#include <dolfin/Matrix.h>
#include <dolfin/Vector.h>
#include <dolfin/Boundary.h>
#include <dolfin/BoundaryValue.h>
#include <dolfin/BoundaryCondition.h>
#include <dolfin/BoundaryFacetIterator.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/GenericMatrix.h>
#include <dolfin/GenericVector.h>
#include <dolfin/BilinearForm.h>
#include <dolfin/LinearForm.h>

namespace dolfin
{

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
    template<class T>
    static void assemble(BilinearForm& a, GenericMatrix<T>& A, Mesh& mesh);

    /// Assemble linear form
    template<class T>
    static void assemble(LinearForm& L, GenericVector<T>& b, Mesh& mesh);

    /// Assemble bilinear and linear forms
    template<class T, class U>
    static void assemble(BilinearForm& a, LinearForm& L, GenericMatrix<T>& A, 
       GenericVector<U>& b, Mesh& mesh);
    
    /// Assemble bilinear and linear forms (including Dirichlet boundary conditions)
    template<class T, class U>
    static void assemble(BilinearForm& a, LinearForm& L,  GenericMatrix<T>& A, 
       GenericVector<U>& b, Mesh& mesh, BoundaryCondition& bc);
    
    /// Apply boundary conditions to matrix and vector 
    template<class T, class U>
    static void applyBC(GenericMatrix<T>& A, GenericVector<U>& b, Mesh& mesh,
			FiniteElement& element, BoundaryCondition& bc);
    
    /// Apply boundary conditions to matrix 
    template<class T>
    static void applyBC(GenericMatrix<T>& A, Mesh& mesh, FiniteElement& element, 
			BoundaryCondition& bc);

    /// Apply boundary conditions to vector. 
    template<class T>
    static void applyBC(GenericVector<T>& b, Mesh& mesh, FiniteElement& element,
			BoundaryCondition& bc);

    /// Assemble boundary conditions into residual vector.  For Dirichlet 
    /// boundary conditions, b = x - bc, and for Neumann boundary 
    /// conditions, b = b - bc.
#ifdef HAVE_PETSC_H
    // FIXME: Should this also be templated over the type for x?
    template<class T>
    static void assembleBCresidual(GenericVector<T>& b, const Vector& x, Mesh& mesh, 
				   FiniteElement& element, BoundaryCondition& bc);
    
    /// Assemble boundary conditions into residual vector.  For Dirichlet 
    /// boundary conditions, b = x - bc, and for Neumann boundary 
    /// conditions, b = b - bc. 
    template<class T, class U>
    static void assembleBCresidual(GenericMatrix<T>& A, GenericVector<U>& b, 
				   const Vector& x, Mesh& mesh, FiniteElement& element, BoundaryCondition& bc);
#endif

    /// Count the degrees of freedom
    static uint size(const Mesh& mesh, const FiniteElement& element);

    /// Lump matrix
#ifdef HAVE_PETSC_H
    // FIXME: Why is this not templated like the rest?
    static void lump(const Matrix& M, Vector& m);
#endif

    static void lump(const DenseMatrix& M, DenseVector& m);

    /// Display assembly data (useful for debugging)
    static void disp(const Mesh& mesh, const FiniteElement& element);
      
  private:

    /// Assemble form(s)
    template<class T, class U>
    static void assemble_common(BilinearForm* a, LinearForm* L, GenericMatrix<T>* A, 
				GenericVector<U>* b, Mesh& mesh);

    /// Create boundary facet interator amnd assemble form(s)
    template<class T, class U, class V, class W>
    static void assemble_common(BilinearForm* a, LinearForm* L, GenericMatrix<T>* A, 
				GenericVector<U>* b, Mesh& mesh, BoundaryFacetIterator<V,W>& facet);

    /// Create iterator and call function to apply boundary conditions
    template<class T, class U>
    static void applyBC_common(GenericMatrix<T>* A, GenericVector<U>* b, const Vector* x,
			       Mesh& mesh, FiniteElement& element, BoundaryCondition& bc);

    /// Apply boundary conditions
    template<class T, class U, class V, class W>
    static void applyBC_common(GenericMatrix<T>* A, GenericVector<W>* b, 
			       const Vector* x, Mesh& mesh, FiniteElement& element, BoundaryCondition& bc, 
			       BoundaryFacetIterator<U,V>& facet);
    
    /// Estimate the maximum number of nonzeros in each row
    static uint nzsize(const Mesh& mesh, const FiniteElement& element);

    /// Check that dimension of the mesh matches the form
    static void checkdims(BilinearForm& a, const Mesh& mesh);

    /// Check that dimension of the mesh matches the form
    static void checkdims(LinearForm& L, const Mesh& mesh);

    /// Check number of nonzeros in each row
    template<class T>
    static void checknz(GenericMatrix<T>& A, uint nz);

  };

  //-----------------------------------------------------------------------------
  // Public member definitions
  //-----------------------------------------------------------------------------
  template<class T>
  void FEM::assemble(BilinearForm& a, GenericMatrix<T>& A, Mesh& mesh)
  {
    assemble_common(&a, 0, &A, (DenseVector*) 0, mesh);
  }
  //-----------------------------------------------------------------------------
  template<class T>
  void FEM::assemble(LinearForm& L, GenericVector<T>& b, Mesh& mesh)
  {
    assemble_common(0, &L, (DenseMatrix*) 0, &b, mesh);
  }
  //-----------------------------------------------------------------------------
  template<class T, class U>
  void FEM::assemble(BilinearForm& a, LinearForm& L, GenericMatrix<T>& A, 
        GenericVector<U>& b, Mesh& mesh)
  {
    assemble_common(&a, &L, &A, &b, mesh);
  }
  //-----------------------------------------------------------------------------
  template<class T, class U>
  void FEM::assemble(BilinearForm& a, LinearForm& L, GenericMatrix<T>& A, 
		     GenericVector<U>& b, Mesh& mesh, BoundaryCondition& bc)
  {
    assemble_common(&a, &L, &A, &b, mesh);
    applyBC(A, b, mesh, a.trial(), bc);
  }
  //-----------------------------------------------------------------------------
  template<class T, class U>
  void FEM::applyBC(GenericMatrix<T>& A, GenericVector<U>& b, Mesh& mesh, 
		    FiniteElement& element, BoundaryCondition& bc)
  {
    dolfin_info("Applying Dirichlet boundary conditions.");
    applyBC_common(&A, &b, 0, mesh, element, bc);
  }
  //-----------------------------------------------------------------------------
  template<class T>
  void FEM::applyBC(GenericMatrix<T>& A, Mesh& mesh, FiniteElement& element, 
		    BoundaryCondition& bc)
  {
    dolfin_info("Applying Dirichlet boundary conditions to matrix.");
    applyBC_common(&A, (DenseVector*) 0, 0, mesh, element, bc);
  }
  //-----------------------------------------------------------------------------
  template<class T>
  void FEM::applyBC(GenericVector<T>& b, Mesh& mesh, FiniteElement& element, BoundaryCondition& bc)
  {
    dolfin_info("Applying Dirichlet boundary conditions to vector.");
    applyBC_common(0, &b, 0, mesh, element, bc);
  }
  //-----------------------------------------------------------------------------
#ifdef HAVE_PETSC_H
    // FIXME: Should this also be templated over the type for x?
  template<class T, class U>
  void FEM::assembleBCresidual(GenericMatrix<T>& A, GenericVector<U>& b, 
			       const Vector& x, Mesh& mesh, FiniteElement& element, BoundaryCondition& bc)
  {
    dolfin_info("Assembling boundary condtions into residual vector.");
    applyBC_common(&A, &b, &x, mesh, element, bc);
  }
  //-----------------------------------------------------------------------------
  template<class T>
  void FEM::assembleBCresidual(GenericVector<T>& b, const Vector& x, Mesh& mesh,
			       FiniteElement& element, BoundaryCondition& bc)
  {
    dolfin_info("Assembling boundary condtions into residual vector.");
    applyBC_common((DenseMatrix*) 0, &b, &x, mesh, element, bc);
  }
#endif
  //-----------------------------------------------------------------------------
  //  Private member definitions
  //-----------------------------------------------------------------------------
  template<class T, class U>
  void FEM::assemble_common(BilinearForm* a, LinearForm* L, GenericMatrix<T>* A, 
			    GenericVector<U>* b, Mesh& mesh)
  {
    // Create boundary
    Boundary boundary(mesh);
    
    // Choose type of mesh
    if( mesh.type() == Mesh::triangles )
    {
      BoundaryFacetIterator<EdgeIterator,Edge> facet(boundary);
      assemble_common(a, L, A, b, mesh, facet);
    }
    else if( mesh.type() == Mesh::tetrahedra )
    {
      BoundaryFacetIterator<FaceIterator,Face> facet(boundary);
      assemble_common(a, L, A, b, mesh, facet);
    }
    else
    {
      dolfin_error("Unknown mesh type.");  
    }
  }
  //-----------------------------------------------------------------------------
  template<class T, class U, class V, class W>
  void FEM::assemble_common(BilinearForm* a, LinearForm* L, GenericMatrix<T>* A, 
			    GenericVector<U>* b, Mesh& mesh, BoundaryFacetIterator<V,W>& facet)
  {
    // Check that the mesh matches the forms
    if( a )
      checkdims(*a, mesh);
    if( L )
      checkdims(*L, mesh);
 
    // Get finite elements
    FiniteElement* test_element  = 0;
    FiniteElement* trial_element = 0;
    if( a )
    {
      test_element  = &(a->test());
      trial_element = &(a->trial());
    }
    else if( L ) 
      test_element = &(L->test());

    // Create affine map
    AffineMap map;

    // Initialize element matrix/vector data block
    real* block_A = 0;
    real* block_b = 0;
    int* test_nodes = 0;
    int* trial_nodes = 0;
    uint n  = 0;
    uint N  = 0;
    uint nz = 0;

    const uint m = test_element->spacedim();
    const uint M = size(mesh, *test_element);
    test_nodes = new int[m];

    if( a )
    {
      n = trial_element->spacedim();
      N = size(mesh, *trial_element);
      block_A = new real[m*n];
      trial_nodes = new int[m];
      nz = nzsize(mesh, *trial_element);
      A->init(M, N, nz);
      *A = 0.0;
    }
    if( L )
    {
      block_b = new real[m];  
      b->init(M);
      *b = 0.0;
    }
    // Start a progress session
    if( a && L)
      dolfin_info("Assembling system (matrix and vector) of size %d x %d.", M, N);
    if( a && !L)
      dolfin_info("Assembling matrix of size %d x %d.", M, N);
    if( !a && L)
      dolfin_info("Assembling vector of size %d.", M);
    Progress p("Assembling interior contributions", mesh.numCells());
   
    // Iterate over all cells in the mesh
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      // Update affine map
      map.update(*cell);
  
      // Compute map from local to global degrees of freedom (test functions)
      test_element->nodemap(test_nodes, *cell, mesh);
  
      if( a )
      {
        // Update forms
        a->update(map);
        // Compute maps from local to global degrees of freedom (trial functions)
        trial_element->nodemap(trial_nodes, *cell, mesh);
        // Compute element matrix 
        a->eval(block_A, map);
        // Add element matrix to global matrix
        A->add(block_A, test_nodes, m, trial_nodes, n);
      }
      if( L )
      {
        // Update forms
        L->update(map);    
        // Compute element vector 
        L->eval(block_b, map);
        // Add element vector to global vector
        b->add(block_b, test_nodes, m);
      }

      // Update progress
      p++;
    }
  
    //FIXME: need to reinitiliase block_A and block_b in case no boudary terms are provided
    if( a )
      for (uint i = 0; i < m*n; ++i)
        block_A[i] = 0.0;
    if( L )
      for (uint i = 0; i < m; ++i)
        block_b[i] = 0.0;

    // Iterate over all facets on the boundary
    Boundary boundary(mesh);
    Progress p_boundary("Assembling boudary contributions", boundary.numFacets());
    for ( ; !facet.end(); ++facet)
    {
      // Get cell containing the edge (pick first, should only be one)
      dolfin_assert(facet.numCellNeighbors() == 1);
      Cell& cell = facet.cell(0);

      // Get local facet ID
      uint facetID = facet.localID(0);
      
      // Update affine map for facet 
      map.update(cell, facetID);
  
      // Compute map from local to global degrees of freedom (test functions)
      test_element->nodemap(test_nodes, cell, mesh);
  
      if( a )
      {
        // Update forms
        a->update(map);  
        // Compute maps from local to global degrees of freedom (trial functions)
        trial_element->nodemap(trial_nodes, cell, mesh);

        // Compute element matrix 
        a->eval(block_A, map, facetID);

        // Add element matrix to global matrix
        A->add(block_A, test_nodes, m, trial_nodes, n);
      }
      if( L )
      {
        // Update forms
        L->update(map);    
        // Compute element vector 
        L->eval(block_b, map, facetID);
 
        // Add element vector to global vector
        b->add(block_b, test_nodes, m);
      }
      // Update progress
      p_boundary++;
    }

    // Complete assembly
    if( L )
      b->apply();
    if ( a )
    {
      A->apply();
      // Check the number of nonzeros
      checknz(*A, nz);
    }
    
    // Delete data
    delete [] block_A;
    delete [] block_b;
    delete [] trial_nodes;
    delete [] test_nodes;
  }
  //-----------------------------------------------------------------------------
#ifdef HAVE_PETSC_H
    // FIXME: Should this also be templated over the type for x?
  template<class T, class U>
  void FEM::applyBC_common(GenericMatrix<T>* A, GenericVector<U>* b, 
			   const Vector* x, Mesh& mesh, FiniteElement& element, BoundaryCondition& bc)
  {
    // Create boundary
    Boundary boundary(mesh);

    // Choose type of mesh
    if( mesh.type() == Mesh::triangles)
    {
      BoundaryFacetIterator<EdgeIterator,Edge> facet(boundary);
      applyBC_common(A, b, x, mesh, element, bc, facet);
    }
    else if(mesh.type() == Mesh::tetrahedra)
    {
      BoundaryFacetIterator<FaceIterator,Face> facet(boundary);
      applyBC_common(A, b, x, mesh, element, bc, facet);
    }
    else
    {
      dolfin_error("Unknown mesh type.");  
    }
  }
  //-----------------------------------------------------------------------------
  template<class T, class U, class V, class W>
  void FEM::applyBC_common(GenericMatrix<T>* A, GenericVector<W>* b, 
			   const Vector* x, Mesh& mesh, FiniteElement& element, 
			   BoundaryCondition& bc, BoundaryFacetIterator<U,V>& facet)
  {
    // Create boundary value
    BoundaryValue bv;

    // Create affine map
    AffineMap map;

    // Compute problem size
    uint size = 0;
    if( A )
      size = A->size(0);
    else
      size = b->size();

    // Allocate list of rows
    uint m = 0;
    int*  rows = 0;
    if ( A )
      rows = new int[size];

    bool* row_set = new bool[size];
    for (unsigned int i = 0; i < size; i++)
      row_set[i] = false;

    // Allocate local data
    uint n = element.spacedim();
    int* nodes = new int[n];
    uint* components = new uint[n];
    Point* points = new Point[n];

    real* block_b = 0;
    int* node_list = 0;
    if( b )
    {
      block_b   = new real[n];  
      node_list = new int[n];  
    }

    // Iterate over all edges/faces on the boundary
    for ( ; !facet.end(); ++facet)
    {
      uint k = 0;

      // Get cell containing the edge (pick first, should only be one)
      dolfin_assert(facet.numCellNeighbors() == 1);
      Cell& cell = facet.cell(0);

      // Update affine map
      map.update(cell);
      // Compute map from local to global degrees of freedom
      element.nodemap(nodes, cell, mesh);
      // Compute map from local to global coordinates
      element.pointmap(points, components, map);

      // Set boundary conditions for nodes on the boundary
      for (uint i = 0; i < n; i++)
      {
        // Skip points that are not contained in edge
        const Point& point = points[i];
        if ( !(facet.contains(point)) )
          continue;

        // Get boundary condition
        bv.reset();
        bc.eval(bv, point, components[i]);

        // Set boundary condition if Dirichlet
        if ( bv.fixed )
        {
          int node = nodes[i];
          if ( !row_set[node] )
          {
            if( x ) // Compute "residual" 
              block_b[k] = bv.value - (*x)(node);
            else if( b ) 
              block_b[k] = bv.value;

            row_set[node] = true;

            if ( b )
              node_list[k++] = node;
            if ( A )
              rows[m] = node;

            m++;
          }
        }
      }
      if( b )
        b->insert(block_b, node_list, k);
    }
    dolfin_info("Boundary condition applied to %d degrees of freedom on the boundary.", m);

    // Set rows of matrix to the identity matrix
    if( A )
      A->ident(rows, m);

    if( b )
      b->apply();

    // Delete data
    delete [] nodes;
    delete [] components;
    delete [] points;
    delete [] rows;
    delete [] row_set;
    delete [] block_b;
    delete [] node_list;
  }
#endif
  //-----------------------------------------------------------------------------
  template <class T>
  void FEM::checknz(GenericMatrix<T>& A, uint nz)
  {
    uint nz_actual = A.nzmax();
    if ( nz_actual > nz )
      dolfin_warning("Actual number of nonzero entries exceeds estimated number of nonzero entries.");
    else
      dolfin_info("Maximum number of nonzeros in each row is %d (estimated %d).",
  		nz_actual, nz);
  }
  //-----------------------------------------------------------------------------
  
}

#endif
