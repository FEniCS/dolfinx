// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-07-03
// Last changed:

#ifndef __NEW_LU_SOLVER_H
#define __NEW_LU_SOLVER_H

#include <dolfin/Parametrized.h>
#include <dolfin/NewLinearSolver.h>
#include <dolfin/NewVector.h>
#include <dolfin/NewMatrix.h>
#include <dolfin/default_la_types.h>

namespace dolfin
{

  class NewLUSolver : public NewLinearSolver, public Parametrized
  {
    /// This class defines an interface for a LU solver. The underlying type of 
    /// LU is defined in default_type.h.
    
    public:

      NewLUSolver(){}

      ~NewLUSolver() {}

      uint solve(const NewMatrix& A, NewVector& x, const NewVector& b)
        { return solver.solve(A.mat(), x.vec(), b.vec()); }

    private:

      DefaultLUSolver solver;

  };
}

#endif
