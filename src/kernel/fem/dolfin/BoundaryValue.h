// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-02-13
// Last changed: 2005

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
    const BoundaryValue& operator= (real value);

    /// Set Dirichlet boundary value
    void set(real value);
    
    /// Friends
    friend class FEM;
    friend class NewFEM;

  private:

    // True if boundary value is fixed (Dirichlet)
    bool fixed;

    // The boundary value
    real value;

  };
  
}

#endif
