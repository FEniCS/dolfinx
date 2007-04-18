// Copyright (C) 2004-2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2006.
//
// First added:  2004
// Last changed: 2007-04-17

#ifndef __LINEAR_PDE_H
#define __LINEAR_PDE_H

#include <dolfin/GenericPDE.h>

namespace dolfin
{

  /// This class implements the solution functionality for linear PDEs.

  class LinearPDE : public GenericPDE
  {
  public:

    /// Constructor
    LinearPDE(Form& a, Form& L, Mesh& mesh, Array<BoundaryCondition*> bcs);

    /// Destructor
    ~LinearPDE();
    
    /// Solve PDE
    void solve(Function& u);

  };

}

#endif
