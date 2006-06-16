// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-02-21
// Last changed: 2006-02-24

#ifndef __GENERIC_PDE_H
#define __GENERIC_PDE_H

#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <dolfin/Parametrized.h>

namespace dolfin
{
  
  class BilinearForm;
  class BoundaryCondition;
  class Function;
  class LinearForm;
  class Mesh;

  /// This class serves as a base class/interface for specific PDE's.

  class GenericPDE : public Parametrized
  {
  public:

    /// Constructor
    GenericPDE();

    /// Destructor
    virtual ~GenericPDE();

     /// Solve
    virtual uint solve(Function& u) = 0;

     /// Return element dimension
    virtual uint elementdim() = 0;

    /// Return the bilinear form mesh associated with PDE (if any)
    virtual BilinearForm& a() = 0;

    /// Return the linear form mesh associated with PDE (if any)
    virtual LinearForm& L() =0;

    /// Return the mesh associated with PDE (if any)
    virtual Mesh& mesh() = 0;

    /// Return the boundary conditions associated with PDE (if any)
    virtual BoundaryCondition& bc() = 0;

  };

}

#endif
