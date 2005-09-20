// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2005.
//
// First added:  2005-05-02
// Last changed: 2005-09-20

#ifndef __BOUNDARY_CONDITION_H
#define __BOUNDARY_CONDITION_H

#include <dolfin/TimeDependent.h>
#include <dolfin/constants.h>
#include <dolfin/BoundaryValue.h>
#include <dolfin/Point.h>

namespace dolfin
{

  class BoundaryCondition : public TimeDependent
  {
  public:
    
    /// Constructors
    BoundaryCondition();
    
    /// Destructor
    virtual ~BoundaryCondition();

    /// User-defined boundary value for given part of boundary: scalar
    virtual const BoundaryValue operator() (const Point& p);
    
    /// User-defined boundary value for given part of boundary: vector component i 
    virtual const BoundaryValue operator() (const Point& p, uint i);

  };
  
}

#endif
