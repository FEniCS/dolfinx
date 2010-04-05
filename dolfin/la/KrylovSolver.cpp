// Copyright (C) 2007-2009 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ola Skavhaug, 2008.
// Modified by Anders Logg, 2008.
//
// First added:  2007-07-03
// Last changed: 2009-10-28

#include <dolfin/common/Timer.h>
#include <dolfin/parameter/Parameters.h>
#include <dolfin/parameter/GlobalParameters.h>
#include "GenericMatrix.h"
#include "GenericVector.h"
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
#include "KrylovSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Parameters KrylovSolver::default_parameters()
{
  Parameters p("krylov_solver");

  p.add("relative_tolerance",  1e-6);
  p.add("absolute_tolerance",  1e-15);
  p.add("divergence_limit",    1e4);
  p.add("maximum_iterations",  10000);
  p.add("gmres_restart",       30);
  p.add("shift_nonzero",       0.0);
  p.add("report",              true); /* deprecate? */
  p.add("monitor_convergence", false);
  return p;
}
//-----------------------------------------------------------------------------
KrylovSolver::KrylovSolver(std::string solver_type, std::string pc_type)
       : solver_type(solver_type), pc_type(pc_type), ublas_solver(0),
         petsc_solver(0), epetra_solver(0), itl_solver(0)
{
  // Set default parameters
  parameters = dolfin::parameters("krylov_solver");
}
//-----------------------------------------------------------------------------
KrylovSolver::~KrylovSolver()
{
  delete ublas_solver;
  delete petsc_solver;
  delete epetra_solver;
  delete itl_solver;
}
//-----------------------------------------------------------------------------
dolfin::uint KrylovSolver::solve(const GenericMatrix& A, GenericVector& x,
                                 const GenericVector& b)
{
  Timer timer("Krylov solver");

  if (A.has_type<uBLASSparseMatrix>())
  {
    if (!ublas_solver)
    {
      ublas_solver = new uBLASKrylovSolver(solver_type, pc_type);
      ublas_solver->parameters.update(parameters);
    }
    return ublas_solver->solve(A, x, b);
  }

  if (A.has_type<uBLASDenseMatrix>())
  {
    if (!ublas_solver)
    {
      ublas_solver = new uBLASKrylovSolver(solver_type, pc_type);
      ublas_solver->parameters.update(parameters);
    }
    return ublas_solver->solve(A.down_cast<uBLASDenseMatrix>(),
                               x.down_cast<uBLASVector>(),
                               b.down_cast<uBLASVector>());
  }

#ifdef HAS_PETSC
  if (A.has_type<PETScMatrix>())
  {
    if (!petsc_solver)
    {
      petsc_solver = new PETScKrylovSolver(solver_type, pc_type);
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
      epetra_solver = new EpetraKrylovSolver(solver_type, pc_type);
      epetra_solver->parameters.update(parameters);
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
      itl_solver->parameters.update(parameters);
    }
    return itl_solver->solve(A, x, b);
  }
#endif

  error("No default Krylov solver for given backend");
  return 0;
}
//-----------------------------------------------------------------------------
