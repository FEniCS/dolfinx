// Copyright (C) 2007-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2008.
// Modified by Dag Lindbo, 2008.
// Modified by Anders Logg, 2008.
// Modified by Kent-Andre Mardal, 2008.
//
// First added:  2007-07-03
// Last changed: 2008-06-13

#ifndef __LU_SOLVER_H
#define __LU_SOLVER_H

#include <dolfin/parameter/Parametrized.h>
#include <dolfin/common/Timer.h>
#include "GenericMatrix.h"
#include "GenericVector.h"
#include "UmfpackLUSolver.h"
#include "uBlasSparseMatrix.h"
#include "uBlasDenseMatrix.h"
#include "PETScLUSolver.h"
#include "PETScMatrix.h"
#include "EpetraLUSolver.h"
#include "EpetraMatrix.h"

namespace dolfin
{

  class LUSolver : public Parametrized
  {

  /// LU solver for the built-in LA backends. 
    
  public:

    LUSolver() : umfpack_solver(0), petsc_solver(0), epetra_solver(0) {}
    
    ~LUSolver() 
    { 
      delete umfpack_solver; 
      delete petsc_solver; 
      delete epetra_solver; 
    }
    
    uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b)
    {
      Timer timer("LU solver");

      if (A.has_type<uBlasSparseMatrix>()) 
      {
        if (!umfpack_solver)
        {
          umfpack_solver = new UmfpackLUSolver();
          umfpack_solver->set("parent", *this);
        }
        return umfpack_solver->solve(A.down_cast<uBlasSparseMatrix>(), x.down_cast<uBlasVector>(), b.down_cast<uBlasVector>());
      }

#ifdef HAS_PETSC
      if (A.has_type<PETScMatrix>()) 
      {
        if (!petsc_solver)
        {
          petsc_solver = new PETScLUSolver();
          petsc_solver->set("parent", *this);
        }
        return petsc_solver->solve(A.down_cast<PETScMatrix>(), x.down_cast<PETScVector>(), b.down_cast<PETScVector>());
      }
#endif
#ifdef HAS_TRILINOS
      if (A.has_type<EpetraMatrix>()) 
      {
        if (!epetra_solver)
        {
          epetra_solver = new EpetraLUSolver();
          epetra_solver->set("parent", *this);
        }
        return epetra_solver->solve(A.down_cast<EpetraMatrix>(), x.down_cast<EpetraVector>(), b.down_cast<EpetraVector>());
      }
#endif

      error("No default LU solver for given backend");
      return 0;
    }

    uint factorize(const GenericMatrix& A)
    {
      if (A.has_type<uBlasSparseMatrix>()) 
      {
        if (!umfpack_solver)
        {
          umfpack_solver = new UmfpackLUSolver();
          umfpack_solver->set("parent", *this);
        }
        return umfpack_solver->factorize(A.down_cast<uBlasSparseMatrix>());
      }

      if (A.has_type<uBlasDenseMatrix>())
        error("Will only factorize sparse matrices");

      error("No matrix factorization for given backend.");
      return 0;
    }
    
    uint factorized_solve(GenericVector& x, const GenericVector& b)
    {
      if (b.has_type<uBlasVector>()) 
      {
        if (!umfpack_solver)
        {
          umfpack_solver = new UmfpackLUSolver();
          umfpack_solver->set("parent", *this);
        }
        return umfpack_solver->factorizedSolve(x.down_cast<uBlasVector>(), b.down_cast<uBlasVector>());
      }

      error("No factorized LU solver for given backend.");
      return 0;
    }

  private:

    // uBLAS solver
    UmfpackLUSolver* umfpack_solver;

    // PETSc Solver
#ifdef HAS_PETSC
    PETScLUSolver* petsc_solver;
#else
    int* petsc_solver;
#endif
#ifdef HAS_TRILINOS
    EpetraLUSolver* epetra_solver;
#else
    int* epetra_solver;
#endif



  };
}

#endif
