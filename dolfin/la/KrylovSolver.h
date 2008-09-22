// Copyright (C) 2007-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2008.
// Modified by Anders Logg, 2008.
//
// First added:  2007-07-03
// Last changed: 2008-08-26

#ifndef __KRYLOV_SOLVER_H
#define __KRYLOV_SOLVER_H

#include <dolfin/parameter/Parametrized.h>
#include <dolfin/common/Timer.h>
#include "GenericMatrix.h"
#include "GenericVector.h"
#include "enums_la.h"
#include "uBLASKrylovSolver.h"
#include "uBLASSparseMatrix.h"
#include "uBLASDenseMatrix.h"
#include "EpetraKrylovSolver.h"
#include "ITLKrylovSolver.h"
#include "MTL4Matrix.h"
#include "MTL4Vector.h"
#include "PETScKrylovSolver.h"
#include "PETScMatrix.h"
#include "PETScVector.h"
#include "EpetraMatrix.h"
#include "EpetraVector.h"

namespace dolfin
{

  /// This class defines an interface for a Krylov solver. The underlying 
  /// Krylov solver type is defined in default_type.h.
  
  class KrylovSolver : public Parametrized
  {
  public:
    
    /// Create Krylov solver
    KrylovSolver(dolfin::SolverType solver_type=default_solver,
                 dolfin::PreconditionerType pc_type=default_pc)
      : solver_type(solver_type), pc_type(pc_type), ublas_solver(0), petsc_solver(0), 
        epetra_solver(0), itl_solver(0) {}
    
    /// Destructor
    ~KrylovSolver()
    {
      delete ublas_solver; 
      delete petsc_solver; 
      delete epetra_solver; 
      delete itl_solver; 
    }
    
    /// Solve linear system Ax = b
    uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b)
    { 
      Timer timer("Krylov solver");

      if (A.has_type<uBLASSparseMatrix>())
      {
        if (!ublas_solver)
        {
          ublas_solver = new uBLASKrylovSolver(solver_type, pc_type);
          ublas_solver->set("parent", *this);
        }
        return ublas_solver->solve(A, x, b);
      }

      if (A.has_type<uBLASDenseMatrix>())
      {
        if (!ublas_solver)
        {
          ublas_solver = new uBLASKrylovSolver(solver_type, pc_type);
          ublas_solver->set("parent", *this);
        }
        return ublas_solver->solve(A.down_cast<uBLASDenseMatrix>(), x.down_cast<uBLASVector>(), b.down_cast<uBLASVector>());
      }

#ifdef HAS_PETSC
      if (A.has_type<PETScMatrix>())
      {
        if (!petsc_solver)
        {
          petsc_solver = new PETScKrylovSolver(solver_type, pc_type);
          petsc_solver->set("parent", *this);
        }
        return petsc_solver->solve(A, x, b);
      }
#endif
#ifdef HAS_TRILINOS
      if (A.has_type<EpetraMatrix>())
      {
        if (!epetra_solver)
        {
          epetra_solver = new EpetraKrylovSolver(solver_type, pc_type);
          epetra_solver->set("parent", *this);
        }
        return epetra_solver->solve(A, x, b);
      }
#endif
#ifdef HAS_MTL4
      if (A.has_type<MTL4Matrix>())
      {
        if (!itl_solver)
        {
          itl_solver = new ITLKrylovSolver(solver_type, pc_type);
          itl_solver->set("parent", *this);
        }
        return itl_solver->solve(A, x, b);
      }
#endif

      error("No default Krylov solver for given backend");
      return 0;
    }
    
  private:
    
    // Krylov method
    SolverType solver_type;
    
    // Preconditioner type
    PreconditionerType pc_type;
    
    // uBLAS solver
    uBLASKrylovSolver* ublas_solver;

    // PETSc solver
#ifdef HAS_PETSC
    PETScKrylovSolver* petsc_solver;
#else
    int* petsc_solver;
#endif
#ifdef HAS_TRILINOS
    EpetraKrylovSolver* epetra_solver;
#else
    int* epetra_solver;
#endif
#ifdef HAS_MTL4
    ITLKrylovSolver* itl_solver;
#else
    int* itl_solver;
#endif
  };
}

#endif
