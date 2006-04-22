// Copyright (C) 2004-2005 Johan Hoffman, Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2006
//
// First added:  2004-05-19
// Last changed: 2006-04-12

#ifndef __FEM_H
#define __FEM_H

#include <dolfin/constants.h>
#include <dolfin/AffineMap.h>
#include <dolfin/Mesh.h>
#include <dolfin/Matrix.h>
#include <dolfin/Vector.h>
#include <dolfin/BoundaryValue.h>
#include <dolfin/BoundaryCondition.h>
#include <dolfin/BoundaryIterator.h>
#include <dolfin/FiniteElement.h>

namespace dolfin
{
  class BilinearForm;
  class LinearForm;

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
    static void assemble(BilinearForm& a, LinearForm& L, Matrix& A, Vector& b, 
       Mesh& mesh);
    
    /// Assemble bilinear and linear forms (including Dirichlet boundary conditions)
    static void assemble(BilinearForm& a, LinearForm& L,  Matrix& A, Vector& b, 
       Mesh& mesh, BoundaryCondition& bc);
    
    /// Apply boundary conditions to matrix and vector 
    static void applyBC(Matrix& A, Vector& b, Mesh& mesh,
			FiniteElement& element, BoundaryCondition& bc);
    
    /// Apply boundary conditions to matrix 
    static void applyBC(Matrix& A, Mesh& mesh, FiniteElement& element, 
			BoundaryCondition& bc);

    /// Apply boundary conditions to vector. 
    static void applyBC(Vector& b, Mesh& mesh, FiniteElement& element,
			BoundaryCondition& bc);

    /// Assemble boundary conditions into residual vector.  For Dirichlet 
    /// boundary conditions, b = x - bc, and for Neumann boundary 
    /// conditions, b = b - bc. 
    static void assembleBCresidual(Vector& b, const Vector& x, Mesh& mesh, 
      FiniteElement& element, BoundaryCondition& bc);

    /// Assemble boundary conditions into residual vector.  For Dirichlet 
    /// boundary conditions, b = x - bc, and for Neumann boundary 
    /// conditions, b = b - bc. 
    static void assembleBCresidual(Matrix& A, Vector& b, const Vector& x, 
      Mesh& mesh, FiniteElement& element, BoundaryCondition& bc);

    /// Count the degrees of freedom
    static uint size(const Mesh& mesh, const FiniteElement& element);

    /// Lump matrix
    static void lump(const Matrix& M, Vector& m);

    /// Display assembly data (useful for debugging)
    static void disp(const Mesh& mesh, const FiniteElement& element);
      
  private:

    /// Assemble form(s)
    static void assemble_test(BilinearForm* a, LinearForm* L, Matrix& A, 
        Vector& b, Mesh& mesh);

    /// Create iterator and call function to apply boundary conditions
    static void applyBC(Matrix* A, Vector* b, const Vector* x,
         Mesh& mesh, FiniteElement& element, BoundaryCondition& bc);

    /// Apply boundary conditions
    template<class U, class V>
    static void applyBC(Matrix* A, Vector* b, const Vector* x,
         Mesh& mesh, FiniteElement& element, BoundaryCondition& bc, 
         BoundaryIterator<U,V>& boundary_entity);

    /// Estimate the maximum number of nonzeros in each row
    static uint nzsize(const Mesh& mesh, const FiniteElement& element);

    /// Check that dimension of the mesh matches the form
    static void checkdims(BilinearForm& a, const Mesh& mesh);

    /// Check that dimension of the mesh matches the form
    static void checkdims(LinearForm& L, const Mesh& mesh);

    /// Check number of nonzeros in each row
    static void checknz(const Matrix& A, uint nz);

  };

//-----------------------------------------------------------------------------
// Definitions
//-----------------------------------------------------------------------------
  template<class U, class V>
  void FEM::applyBC(Matrix* A, Vector* b, const Vector* x, Mesh& mesh, 
          FiniteElement& element, BoundaryCondition& bc, 
          BoundaryIterator<U,V>& boundary_entity)
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
    int*  rows = new int[size];
    bool* row_set = new bool[size];
    for (unsigned int i = 0; i < size; i++)
      row_set[i] = false;
  
    // Allocate local data
    uint n = element.spacedim();
    int* nodes = new int[n];
    uint* components = new uint[n];
    Point* points = new Point[n];
  
    // Iterate over all edges/faces on the boundary
    for ( ; !boundary_entity.end(); ++boundary_entity)
    {
      // Get cell containing the edge (pick first, should only be one)
      dolfin_assert(boundary_entity.numCellNeighbors() == 1);
      Cell& cell = boundary_entity.cell(0);

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
        if ( !(boundary_entity.contains(point)) )
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
            rows[m++] = node;
            // FIXME: Entries into b should be done block-wise  
            if( x ) // Compute "residual" 
              (*b)(node) = bv.value - (*x)(node);
            else if( b ) 
              (*b)(node) = bv.value;
            row_set[node] = true;
          }
        }
      }
    }
    dolfin_info("Boundary condition applied to %d degrees of freedom on the boundary.", m);

    // Set rows of matrix to the identity matrix
    if( A )
      A->ident(rows, m);

    // Delete data
    delete [] nodes;
    delete [] components;
    delete [] points;
    delete [] rows;
    delete [] row_set;
  }
//-----------------------------------------------------------------------------
}

#endif
