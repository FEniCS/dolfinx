// Copyright (C) 2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Dag Lindbo, 2008.
// Modified by Anders Logg, 2008.
//
// First added:  2008-07-16
// Last changed: 2008-08-25

#ifdef HAS_MTL4

#ifndef __ITL_KRYLOV_SOLVER_H
#define __ITL_KRYLOV_SOLVER_H

#include <dolfin/common/types.h>
#include <dolfin/parameter/Parametrized.h>
#include "enums_la.h"

namespace dolfin 
{

  /// Forward declarations
  class MTL4Matrix;
  class MTL4Vector;

  /// This class implements Krylov methods for linear systems
  /// of the form Ax = b. It is a wrapper for the Krylov solvers
  /// of ITL.
  
  class ITLKrylovSolver : public Parametrized
  {
  public:

    /// Create Krylov solver for a particular method and preconditioner
    ITLKrylovSolver(dolfin::SolverType method=default_solver,
                    dolfin::PreconditionerType pc=default_pc);

    /// Create Krylov solver for a particular method and EpetraPreconditioner
    //ITLKrylovSolver(dolfin::SolverType method, ITLPreconditioner& prec);

    /// Destructor
    ~ITLKrylovSolver();

    /// Solve linear system Ax = b and return number of iterations
    uint solve(const MTL4Matrix& A, MTL4Vector& x, const MTL4Vector& b);
    
    /// Display solver data
    void disp() const;

  private: 

    SolverType         method; 
    PreconditionerType pc_type; 
    //ITLPreconditioner* prec; 

  };

}

#endif

#endif
