// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __NEW_SOLVER_H
#define __NEW_SOLVER_H

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_math.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/File.h>
#include <dolfin/Mesh.h>
#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <dolfin/GMRES.h>
#include <dolfin/NewFunction.h>
#include <dolfin/NewBoundaryCondition.h>
#include <dolfin/NewFEM.h>
#include <dolfin/BoundaryValue.h>
#include <dolfin/BilinearForm.h>
#include <dolfin/LinearForm.h>

namespace dolfin
{

  /// This is the base class for all solvers. Since all solvers define
  /// their own interfaces, this class is just a convenient way of
  /// including all the basic functionality that is often needed to
  /// implement a solver.
  
  class NewSolver
  {
  public:

    NewSolver() {}
    
    virtual ~NewSolver() {}

  };

}

#endif
