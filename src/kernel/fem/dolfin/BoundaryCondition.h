// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2005.
//
// First added:  2005-05-02
// Last changed: 2005-12-28

#ifndef __BOUNDARY_CONDITION_H
#define __BOUNDARY_CONDITION_H

#include <dolfin/constants.h>
#include <dolfin/TimeDependent.h>

namespace dolfin
{

  class Point;
  class BoundaryValue;

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

  };
  
}

#endif
