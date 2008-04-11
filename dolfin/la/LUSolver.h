// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug 2008.
//
// First added:  2007-07-03
// Last changed: 2008-04-11

#ifndef __LU_SOLVER_H
#define __LU_SOLVER_H

#include <dolfin/parameter/Parametrized.h>
#include "LinearSolver.h"
#include "Vector.h"
#include "Matrix.h"
#include "default_la_types.h"

namespace dolfin
{

  class LUSolver : public LinearSolver, public Parametrized
  {
    /// This class defines an interface for a LU solver. The underlying type of 
    /// LU is defined in default_la_types.h.
    
  public:

    LUSolver(){}
    
    ~LUSolver() {}
    
    uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b)
    { 
      const DefaultMatrix* AA = dynamic_cast<const DefaultMatrix*>(A.instance()); 
      if (!AA) 
        error("Could not convert first argument to correct backend");
      DefaultVector* xx = dynamic_cast<DefaultVector*>(x.instance()); 
      if (!xx) 
        error("Could not convert second argument to correct backend");
      const DefaultVector* bb = dynamic_cast<const DefaultVector*>(b.instance()); 
      if (!bb)
        error("Could not convert third argument to correct backend");
      return solver.solve(*AA, *xx, *bb); 
    }
    
  private:
    
    DefaultLUSolver solver;
    
  };
}

#endif
