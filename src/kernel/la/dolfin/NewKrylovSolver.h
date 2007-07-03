// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-07-03
// Last changed:

#ifndef __NEW_KRYLOV_SOLVER_H
#define __NEW_KRYLOV_SOLVER_H

#include <dolfin/NewLinearSolver.h>
#include <dolfin/NewVector.h>
#include <dolfin/NewMatrix.h>

#include <dolfin/default_la_types.h>

namespace dolfin
{

  class NewKrylovSolver : public NewLinearSolver, public Variable
  {
    /// This class defines an interface for a Krylov solver. The underlying 
    /// type of Krylov solver is defined in default_type.h.
    
    public:

      NewKrylovSolver(){}

      ~NewKrylovSolver() {}

      uint solve(const NewMatrix& A, NewVector& x, const NewVector& b)
        { return solver.solve(A.mat(), x.vec(), b.vec()); }

    private:

      DefaultKrylovSolver solver;
  };
}

#endif
