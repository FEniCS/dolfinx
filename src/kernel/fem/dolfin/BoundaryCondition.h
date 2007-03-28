// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2005, 2007.
//
// First added:  2005-05-02
// Last changed: 2007-03-22

#ifndef __BOUNDARY_CONDITION_H
#define __BOUNDARY_CONDITION_H

#include <dolfin/constants.h>
#include <dolfin/TimeDependent.h>
#include <ufc.h>

// Forward declarations
class ufc::finite_element;
class ufc::dof_map;

namespace dolfin
{

  // Forward declarations
  class BoundaryValue;
  class Point;
  class GenericMatrix;
  class GenericVector;
  class Mesh;

  /// This class specifies the interface for boundary conditions for
  /// partial differential equations. To specify a boundary condition,
  /// a user must create a subclass of BoundaryCondition and overload
  /// the eval() function, specifying a boundary value as function of 
  /// the coordinates of degrees of freedom on the boundary.

  class BoundaryCondition : public TimeDependent
  {
  public:
    
    /// Constructor
    BoundaryCondition();
    
    /// Destructor
    virtual ~BoundaryCondition();

    /// Evaluate boundary condition at given point p and component i
    virtual void eval(BoundaryValue& value, const Point& p, uint i) = 0;

    /// Apply boundary conditions to a matrix and a vector
    void applyBC(GenericMatrix& A, GenericVector& b, Mesh& mesh, 
                  ufc::finite_element& element);

    /// Apply boundary conditions to a matrix
    void applyBC(GenericMatrix& A, Mesh& mesh, ufc::finite_element& element,
                 ufc::dof_map& dof_map);

    /// Apply boundary conditions to vector
    void applyBC(GenericVector& b, Mesh& mesh, ufc::finite_element& element);

  private:

    /// Apply boundary conditions
    void apply(GenericMatrix* A, GenericVector* b, const GenericVector* x, 
             Mesh& mesh, ufc::finite_element& element, ufc::dof_map& dof_map);

  };
  
}

#endif
