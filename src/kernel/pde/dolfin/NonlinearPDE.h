// Copyright (C) 2005-2006 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2007
//
// First added:  2005-10-24
// Last changed: 2007-04-27

#ifndef __NONLINEAR_PDE_H
#define __NONLINEAR_PDE_H

#include <dolfin/GenericPDE.h>
#include <dolfin/NonlinearProblem.h>

namespace dolfin
{

  /// This class implements the solution functionality for nonlinear PDEs.
  
  class NonlinearPDE : public GenericPDE, public NonlinearProblem
  {
  public:

    /// Constructor
    NonlinearPDE(Form& a, Form& L, Mesh& mesh, Array<BoundaryCondition*>& bcs);

    /// Destructor
    ~NonlinearPDE();
    
    /// Solve PDE
    void solve(Function& u);

  };

}

#endif
