// Copyright (C) 2007-2009 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2008.
// Modified by Dag Lindbo, 2008.
// Modified by Anders Logg, 2008-2009.
// Modified by Kent-Andre Mardal, 2008.
//
// First added:  2007-07-03
// Last changed: 2009-07-07

#ifndef __LU_SOLVER_H
#define __LU_SOLVER_H

#include <boost/scoped_ptr.hpp>
#include <dolfin/common/Timer.h>
#include "GenericMatrix.h"
#include "GenericVector.h"
#include "CholmodCholeskySolver.h"
#include "UmfpackLUSolver.h"
#include "uBLASSparseMatrix.h"
#include "uBLASDenseMatrix.h"
#include "PETScLUSolver.h"
#include "PETScMatrix.h"
#include "EpetraLUSolver.h"
#include "EpetraMatrix.h"
#include "MTL4Matrix.h"
#include "MTL4Vector.h"
#include "GenericLinearSolver.h"

namespace dolfin
{

  class LUSolver : public GenericLinearSolver
  {

  /// LU solver for the built-in LA backends. The type can be "lu" or
  /// "cholesky". Cholesky is suitable only for symmetric positive-definite
  /// matrices. Cholesky is not yet suppprted for all backends (which will
  /// default to LU.

  public:

    LUSolver(std::string type = "lu") : cholmod_solver(0),
             petsc_solver(0), epetra_solver(0),
             type(type)
    {
      // Set default parameters
      parameters = default_parameters();
    }

    ~LUSolver()
    {
      delete cholmod_solver;
      delete petsc_solver;
      delete epetra_solver;
    }

    uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b)
    {
      Timer timer("LU solver");

#ifdef HAS_PETSC
      if (A.has_type<PETScMatrix>())
      {
        if (!petsc_solver)
        {
          petsc_solver = new PETScLUSolver();
          petsc_solver->parameters.update(parameters);
        }
        return petsc_solver->solve(A, x, b);
      }
#endif
#ifdef HAS_TRILINOS
      if (A.has_type<EpetraMatrix>())
      {
        if (!epetra_solver)
        {
          epetra_solver = new EpetraLUSolver();
          epetra_solver->parameters.update(parameters);
        }
        return epetra_solver->solve(A, x, b);
      }
#endif

      // Default LU solvers
      if (type == "cholesky")
      {
        if (!cholmod_solver)
        {
          cholmod_solver = new CholmodCholeskySolver();
          cholmod_solver->parameters.update(parameters);
        }
        return cholmod_solver->solve(A, x, b);
      }
      else if (type == "lu")
      {
        UmfpackLUSolver solver(A);
        solver.parameters.update(parameters);
        return solver.solve(x, b);
      }
      else
      {
        error("Unknown LU solver type %s. Options are \"cholesky\" or \"lu\".", type.c_str());
        return 0;
      }
    }

    /*
    uint factorize(const GenericMatrix& A)
    {
    if (!umfpack_solver)
        {
          umfpack_solver = new UmfpackLUSolver();
          umfpack_solver->parameters.update(parameters);
        }
        return umfpack_solver->factorize(A);
    }

    uint factorized_solve(GenericVector& x, const GenericVector& b)
    {
      if (!umfpack_solver)
      {
        umfpack_solver = new UmfpackLUSolver();
        umfpack_solver->parameters.update(parameters);
      }
      return umfpack_solver->factorized_solve(x, b);
    }
    */

    /// Default parameter values
    static Parameters default_parameters()
    {
      Parameters p("lu_solver");
      p.add("report", true);
      p.add("same_nonzero_pattern", false);
      return p;
    }

  private:

    // CHOLMOD solver
    CholmodCholeskySolver* cholmod_solver;

    // UMFPACK solver
    boost::scoped_ptr<UmfpackLUSolver> umfpack_solver;

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

    std::string type;

  };
}

#endif
