// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __BOUNDARY_VALUE_H
#define __BOUNDARY_VALUE_H

#include <dolfin/constants.h>

namespace dolfin
{

  class BoundaryValue
  {
  public:
    
    /// Create default boundary value (homogeneous Neumann)
    BoundaryValue();

    /// Destructor
    ~BoundaryValue();

    /// Set Dirichlet boundary value
    void set(real value);
    
    /// Friends
    friend class NewFEM;

  private:

    // True if boundary value is fixed (Dirichlet)
    bool fixed;

    // The boundary value
    real value;

  };
  
}

#endif
