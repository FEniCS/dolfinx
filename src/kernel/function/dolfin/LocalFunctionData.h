// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-11-26
// Last changed: 2006-05-30

#ifndef __LOCAL_FUNCTION_DATA_H
#define __LOCAL_FUNCTION_DATA_H

#include <dolfin/constants.h>

namespace dolfin
{
  class FiniteElement;
  class Point;

  /// Class collects containers for local storage used for interpolation
  /// evaluation of functions on local elements.

  class LocalFunctionData
  {
  public:
    
    /// Constructor
    LocalFunctionData();
    
    /// Destructor
    ~LocalFunctionData();
    
    /// Initialize data for given element
    void init(const FiniteElement& element);
    
    /// Clear data
    void clear();
    
    /// Global numbers for local degrees of freedom
    int* dofs;

    /// Component indices for local degrees of freedom
    uint* components;

    /// Interpolation points for local degrees of freedom
    Point* points;

    /// Vertex nodes for components of function
    uint* vertex_nodes;

    /// Local storage for coefficients on a cell 
    real* coefficients;

  private:
    
    // Dimension of local function space
    unsigned int n;
    
  };
  
}

#endif
