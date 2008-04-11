// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug 2008.
//
// First added:  2007-07-03
// Last changed: 2008-04-11

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
    
    inline uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b)
    { 
      const DefaultMatrix* AA = dynamic_cast<const DefaultMatrix*>(A.instance()); 
      if (!AA) error("Could not convert first argument to correct backend");
      DefaultVector* xx = dynamic_cast<DefaultVector*>(x.instance()); 
      if (!xx) error("Could not convert second argument to correct backend");
      const DefaultVector* bb = dynamic_cast<const DefaultVector*>(b.instance()); 
      if (!bb) error("Could not convert third argument to correct backend");
      return solver.solve(*AA, *xx, *bb); 
    }
    
  private:
    
    DefaultKrylovSolver solver;

  };

}

#endif
