// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-07-03
// Last changed: 2007-07-11

#ifndef __KRYLOV_SOLVER_H
#define __KRYLOV_SOLVER_H

#include "LinearSolver.h"
#include <dolfin/parameter/Parametrized.h>
#include "Vector.h"
#include "Matrix.h"

#include "KrylovMethod.h"
#include "Preconditioner.h"

#include "default_la_types.h"

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
    { return solver.solve(*(A.instance()), *(x.instance()), *(b.instance())); }
    
  private:
    
    DefaultKrylovSolver solver;

  };

}

#endif
