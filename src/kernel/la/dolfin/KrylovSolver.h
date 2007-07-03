// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-07-03
// Last changed:

#ifndef __KRYLOV_SOLVER_H
#define __KRYLOV_SOLVER_H

#include <dolfin/LinearSolver.h>
#include <dolfin/Parametrized.h>
#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>

#include <dolfin/KrylovMethod.h>
#include <dolfin/Preconditioner.h>

#include <dolfin/default_la_types.h>

namespace dolfin
{

  class KrylovSolver : public LinearSolver, public Parametrized
  {
    /// This class defines an interface for a Krylov solver. The underlying 
    /// Krylov solver type is defined in default_type.h.
    
    public:

      KrylovSolver(KrylovMethod method = default_method, 
                      Preconditioner pc = default_pc) : LinearSolver(),
                      solver(default_method, default_pc) {}

      ~KrylovSolver() {}

      uint solve(const Matrix& A, Vector& x, const Vector& b)
        { return solver.solve(A.mat(), x.vec(), b.vec()); }

    private:

      DefaultKrylovSolver solver;
  };
}

#endif
