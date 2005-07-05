// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-05-02
// Last changed: 2005

#ifndef __BOUNDARY_CONDITION_H
#define __BOUNDARY_CONDITION_H

#include <dolfin/constants.h>
#include <dolfin/BoundaryValue.h>
#include <dolfin/Point.h>

namespace dolfin
{

  class BoundaryCondition
  {
  public:
    
    /// Constructors
    BoundaryCondition(uint num_components);
    BoundaryCondition();
    
    /// Destructor
    virtual ~BoundaryCondition();

    /// User-defined boundary value for given part of boundary: scalar
    virtual const BoundaryValue operator() (const Point& p);
    
    /// User-defined boundary value for given part of boundary: vector component i 
    virtual const BoundaryValue operator() (const Point& p, const int i);

    /// Return number of components (scalar = 1, vector > 1)
    uint numComponents() const;
    
  private:

    uint num_components;

  };
  
}

#endif
