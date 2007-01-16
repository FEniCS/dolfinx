// Copyright (C) 2006 Garth N. Wels.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2007.

// First added:  2006-12-05
// Last changed: 2007-01-16

#ifndef __DOF_MAP_H
#define __DOF_MAP_H

#include <set>
#include <dolfin/AdjacencyGraph.h>

#include <dolfin/constants.h>
#include <dolfin/Array.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/Mesh.h>

namespace dolfin
{

  // Forward declarations
  class Cell;
  
  /// This class handles degree of freedom mappings. Its constructor takes
  /// the mesh and one or two finite elements. element_0 maps to a vector
  /// or a matrix row, and element_1 maps to the columns of a matrix
  /// Two finite elements are required to generate the sparsity pattern for 
  /// matrix assembly.

  class DofMap
  {
  public:

    // FIXME: When UFC is in place, the DofMap constructor should takes as arguments
    // FIXME: one or more ufc::dof_map
    
    /// Constructor
    DofMap(Mesh& mesh, const FiniteElement* element_0 = 0, const FiniteElement* element_1 = 0);

    /// Destructor
    ~DofMap();

    // FIXME: Remove this function and require elements in constructor

    /// Attach finite elements
    void attach(const FiniteElement* element_0, const FiniteElement* element_1 = 0);

    // FIXME: Rename to map

    /// Get dof map for a cell for element e (e=0 or e=1)
    void dofmap(int dof_map[], const Cell& cell, const uint e = 0) const;

//    /// Return global dof map for element e (e=0 or e=1)
//    const Array< Array<int> >& getMap(const uint e = 0) const;

    /// Return total number of degrees of freedom associated with element e (e=0 or e=1)
    const uint size(const uint e = 0);

    /// Compute number of non-zeroes for a sparse matrix
    uint numNonZeroes();

    /// Compute maximum number of non-zeroes for a row of sparse matrix
    uint numNonZeroesRowMax();

    /// Compute number of non-zeroes for each row in a sparse matrix
    void numNonZeroesRow(int nz_row[]);

  private:

    /// Build complete dof mapping for a given element (e=0 or e=1) on mesh 
    void build(const uint e = 0);

    /// Compute sparsity pattern for a vector for element e (e=0 or e=1)
    void computeVectorSparsityPattern(const uint e = 0);    

    /// Compute sparsity pattern for a matrix where element_0 map to rows 
    /// (usually the test element) and element_1 maps to columns (usually
    //.  the trial element)
    void computeMatrixSparsityPattern();    

    /// Create data layout for compressed storage (CSR) 
    void createCSRLayout();    

    /// Compute adjacency graph
    void computeAdjacencyGraph();    

    // Mesh associated with dof mapping
    Mesh* mesh;

    // Finite elements associated with dof mapping
    const FiniteElement* element[2];

    // Finite element associated with dof mapping (will be ufc::dof_map)
    const FiniteElement* finite_element;

    // Number of degrees of freedom associated with each element
    int _size[2];

    // Degree of freedom map
    Array< Array<int> > map[2];    

    // Vector sparsity pattern represented as a set of nonzero positions
    std::set<int> vector_sparsity_pattern;    

    // Matrix sparsity pattern represented as an Array of set of nonzero 
    // positions. Each set corresponds to a matrix row
    Array< std::set<int> > matrix_sparsity_pattern;    

    // Adjacency graph
    AdjacencyGraph* adjacency_graph;

  };

}

#endif
