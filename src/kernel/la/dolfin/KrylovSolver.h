// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-07-03
// Last changed: 2007-07-11

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

  /// This class defines an interface for a Krylov solver. The underlying 
  /// Krylov solver type is defined in default_type.h.

  class KrylovSolver : public LinearSolver, public Parametrized
  {
  public:

    KrylovSolver() : solver() {}
    
    KrylovSolver(KrylovMethod method) : solver(method, default_pc) {}
    
    KrylovSolver(KrylovMethod method, Preconditioner pc) 
      : solver(method, pc) {}
    
    ~KrylovSolver() {}
    
    inline uint solve(const Matrix& A, Vector& x, const Vector& b)
    { return solver.solve(A.mat(), x.vec(), b.vec()); }
    
  private:
    
    DefaultKrylovSolver solver;

  };

}

#endif
