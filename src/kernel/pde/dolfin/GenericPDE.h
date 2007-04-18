// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2007.
//
// First added:  2006-02-21
// Last changed: 2007-04-17

#ifndef __GENERIC_PDE_H
#define __GENERIC_PDE_H

#include <dolfin/Array.h>
#include <dolfin/Vector.h>
#include <dolfin/Parametrized.h>

namespace dolfin
{

  class Form;
  class Mesh;
  class BoundaryCondition;
  class Function;

  /// This class serves as a base class/interface for specific PDE's.

  class GenericPDE : public Parametrized
  {
  public:

    /// Constructor
    GenericPDE(Form& a, Form& L, Mesh& mesh, Array<BoundaryCondition*> bcs);

    /// Destructor
    virtual ~GenericPDE();

    /// Solve PDE
    virtual void solve(Function& u) = 0;

  protected:

    // The bilinear form
    Form& a;
    
    // The linear form
    Form& L;

    // The mesh
    Mesh& mesh;

    // The boundary conditions
    Array<BoundaryCondition*> bcs;

    // The solution vector
    Vector x;

  };

}

#endif
