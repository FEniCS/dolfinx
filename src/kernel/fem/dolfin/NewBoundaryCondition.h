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
    
    /// Constructors
    NewBoundaryCondition(int no_comp);
    NewBoundaryCondition();
    
    /// Destructor
    virtual ~NewBoundaryCondition();

    /// User-defined boundary value for given part of boundary: scalar
    virtual const BoundaryValue operator() (const Point& p);
    
    /// User-defined boundary value for given part of boundary: vector component i 
    virtual const BoundaryValue operator() (const Point& p, const int i);

    /// Return number of components (scalar = 1, vector > 1)
    int noComp();

  private:

    int no_comp;

  };
  
}

#endif
