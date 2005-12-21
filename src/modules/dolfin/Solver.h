// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-05-02
// Last changed: 2005-12-20

#ifndef __SOLVER_H
#define __SOLVER_H

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_math.h>
#include <dolfin/ParameterSystem.h>
#include <dolfin/Parametrized.h>
#include <dolfin/File.h>
#include <dolfin/Mesh.h>
#include <dolfin/UnitSquare.h>
#include <dolfin/UnitCube.h>
#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <dolfin/KrylovSolver.h>
#include <dolfin/GMRES.h>
#include <dolfin/LU.h>
#include <dolfin/Function.h>
#include <dolfin/BoundaryCondition.h>
#include <dolfin/FEM.h>
#include <dolfin/BoundaryValue.h>
#include <dolfin/BilinearForm.h>
#include <dolfin/LinearForm.h>

namespace dolfin
{

  /// This is the base class for all solvers. Since all solvers define
  /// their own interfaces, this class is just a convenient way of
  /// including all the basic functionality that is often needed to
  /// implement a solver.
  
  class Solver : public Parametrized
  {
  public:

    Solver() {}
    
    virtual ~Solver() {}

  };

}

#endif
