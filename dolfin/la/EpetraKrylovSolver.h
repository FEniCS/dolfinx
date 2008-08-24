// Copyright (C) 2008 Kent-Andre Mardal.
// Licensed under the GNU LGPL Version 2.1.
//
// Last changed: 2008-05-16

#ifdef HAS_TRILINOS

#ifndef __EPETRA_KRYLOV_SOLVER_H
#define __EPETRA_KRYLOV_SOLVER_H

#include <dolfin/common/types.h>
#include <dolfin/parameter/Parametrized.h>
#include "enums_la.h"
#include "EpetraPreconditioner.h"

namespace dolfin 
{

  /// Forward declarations
  class EpetraMatrix;
  class EpetraVector;
  class EpetraKrylovMatrix;

  /// This class implements Krylov methods for linear systems
  /// of the form Ax = b. It is a wrapper for the Krylov solvers
  /// of Epetra.
  
  class EpetraKrylovSolver : public Parametrized
  {
  public:

    /// Create Krylov solver for a particular method and preconditioner
    EpetraKrylovSolver(SolverType method=default_solver, PreconditionerType pc=default_pc);

    /// Create Krylov solver for a particular method and EpetraPreconditioner
    EpetraKrylovSolver(SolverType method, EpetraPreconditioner& prec);

    /// Destructor
    ~EpetraKrylovSolver();

    /// Solve linear system Ax = b and return number of iterations
    uint solve(const EpetraMatrix& A, EpetraVector& x, const EpetraVector& b);
    
    /// Display solver data
    void disp() const;
  private: 
    SolverType         method; 
    PreconditionerType pc_type; 
    EpetraPreconditioner* prec; 


  };

}


#endif 

#endif 


