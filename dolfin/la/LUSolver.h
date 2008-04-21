// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug 2008.
//
// First added:  2007-07-03
// Last changed: 2008-04-16

#ifndef __LU_SOLVER_H
#define __LU_SOLVER_H

#include <dolfin/parameter/Parametrized.h>
#include "LinearSolver.h"
#include "uBlasSparseMatrix.h"
#include "uBlasDenseMatrix.h"
#include "PETScMatrix.h"

namespace dolfin
{

  class LUSolver : public LinearSolver, public Parametrized
  {
    /// LU solver for the built-in LA backends. 
    
  public:

    LUSolver() : 
      ublassolver(0)
#ifdef HAS_PETSC
      , petscsolver(0) 
#endif
    {}
    
    ~LUSolver() { 
      delete ublassolver; 
#ifdef HAS_PETSC
      delete petscsolver; 
#endif
    }
    
    uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b)
    { 
      if (A.has_type<uBlasSparseMatrix>()) {
        if (!ublassolver)
          ublassolver = new uBlasLUSolver();
        return ublassolver->solve(A.down_cast<uBlasSparseMatrix>(), x.down_cast<uBlasVector>(), b.down_cast<uBlasVector>());
      }

      if (A.has_type<uBlasDenseMatrix>()) {
        if (!ublassolver)
          ublassolver = new uBlasLUSolver();
        return ublassolver->solve(A.down_cast<uBlasDenseMatrix >(), x.down_cast<uBlasVector>(), b.down_cast<uBlasVector>());
      }

#ifdef HAS_PETSC
      if (A.has_type<PETScMatrix>()) {
        if (!petscsolver)
          petscsolver = new PETScLUSolver();
        return petscsolver->solve(A.down_cast<PETScMatrix >(), x.down_cast<PETScVector>(), b.down_cast<PETScVector>());
      }
#endif
      error("No default LU solver for given backend");
      return 0;
    }
    
  private:

      uBlasLUSolver* ublassolver;
#ifdef HAS_PETSC
      PETScLUSolver* petscsolver;
#endif
  };
}

#endif
