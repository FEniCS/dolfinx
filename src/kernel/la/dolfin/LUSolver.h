// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-07-03
// Last changed:

#ifndef __LU_SOLVER_H
#define __LU_SOLVER_H

#include <dolfin/Parametrized.h>
#include <dolfin/LinearSolver.h>
#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <dolfin/default_la_types.h>

namespace dolfin
{

  class LUSolver : public LinearSolver, public Parametrized
  {
    /// This class defines an interface for a LU solver. The underlying type of 
    /// LU is defined in default_type.h.
    
    public:

      LUSolver(){}

      ~LUSolver() {}

      uint solve(const Matrix& A, Vector& x, const Vector& b)
        { return solver.solve(A.mat(), x.vec(), b.vec()); }

    private:

      DefaultLUSolver solver;

  };
}

#endif
