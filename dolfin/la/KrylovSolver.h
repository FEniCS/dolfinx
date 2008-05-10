// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2008.
// Modified by Anders Logg, 2008.
//
// First added:  2007-07-03
// Last changed: 2008-05-09

#ifndef __KRYLOV_SOLVER_H
#define __KRYLOV_SOLVER_H

#include <dolfin/parameter/Parametrized.h>
#include "GenericMatrix.h"
#include "GenericVector.h"
#include "uBlasKrylovSolver.h"
#include "uBlasSparseMatrix.h"
#include "uBlasDenseMatrix.h"
#include "PETScKrylovSolver.h"
#include "SolverType.h"
#include "PreconditionerType.h"

namespace dolfin
{

  /// This class defines an interface for a Krylov solver. The underlying 
  /// Krylov solver type is defined in default_type.h.
  
  class KrylovSolver : public Parametrized
  {
  public:
    
    /// Create Krylov solver
    KrylovSolver(SolverType solver_type=default_solver, PreconditionerType pc_type=default_pc)
      : solver_type(solver_type), pc_type(pc_type), ublas_solver(0), petsc_solver(0) {}
    
    /// Destructor
    ~KrylovSolver()
    {
      delete ublas_solver; 
      delete petsc_solver; 
    }
    
    /// Solve linear system Ax = b
    uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b)
    { 
      if (A.has_type<uBlasSparseMatrix>())
      {
        if (!ublas_solver)
        {
          ublas_solver = new uBlasKrylovSolver(solver_type, pc_type);
          ublas_solver->set("parent", *this);
        }
        return ublas_solver->solve(A.down_cast<uBlasSparseMatrix>(), x.down_cast<uBlasVector>(), b.down_cast<uBlasVector>());
      }

      if (A.has_type<uBlasDenseMatrix>())
      {
        if (!ublas_solver)
        {
          ublas_solver = new uBlasKrylovSolver(solver_type, pc_type);
          ublas_solver->set("parent", *this);
        }
        return ublas_solver->solve(A.down_cast<uBlasDenseMatrix>(), x.down_cast<uBlasVector>(), b.down_cast<uBlasVector>());
      }

#ifdef HAS_PETSC
      if (A.has_type<PETScMatrix>())
      {
        if (!petsc_solver)
        {
          petsc_solver = new PETScKrylovSolver(solver_type, pc_type);
          petsc_solver->set("parent", *this);
        }
        return petsc_solver->solve(A.down_cast<PETScMatrix >(), x.down_cast<PETScVector>(), b.down_cast<PETScVector>());
      }
#endif

      error("No default LU solver for given backend");
      return 0;
    }
    
  private:
    
    // Krylov method
    SolverType solver_type;
    
    // Preconditioner type
    PreconditionerType pc_type;
    
    // uBLAS solver
    uBlasKrylovSolver* ublas_solver;

    // PETSc solver
#ifdef HAS_PETSC
    PETScKrylovSolver* petsc_solver;
#else
    int* petsc_solver;
#endif
    
  };

}

#endif
