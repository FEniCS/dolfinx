// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEW_BOUNDARY_CONDITION_H
#define __NEW_BOUNDARY_CONDITION_H

#include <dolfin/constants.h>
#include <dolfin/BoundaryValue.h>
#include <dolfin/Point.h>

namespace dolfin
{

  class NewBoundaryCondition
  {
  public:
    
    /// Constructor
    NewBoundaryCondition();

    /// Destructor
    virtual ~NewBoundaryCondition();

    /// User-defined boundary value for given part of boundary
    virtual const BoundaryValue operator() (const Point& p);

  };
  
}

#endif
