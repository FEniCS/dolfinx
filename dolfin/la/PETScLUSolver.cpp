// Copyright (C) 2005-2011 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells, 2009-2010.
// Modified by Niclas Jansson, 2009.
// Modified by Fredrik Valdmanis, 2011
//
// First added:  2005
// Last changed: 2011-10-06

#ifdef HAS_PETSC

#include <dolfin/common/Timer.h>

#include <boost/assign/list_of.hpp>
#include <dolfin/common/constants.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/MPI.h>
#include <dolfin/parameter/GlobalParameters.h>
#include "LUSolver.h"
#include "PETScMatrix.h"
#include "PETScVector.h"
#include "PETScLUSolver.h"

using namespace dolfin;

// Utility function
namespace dolfin
{
  class PETScKSPDeleter
  {
  public:
    void operator() (KSP* ksp)
    {
      if (ksp)
        #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR <= 1
        KSPDestroy(*ksp);
        #else
        KSPDestroy(ksp);
        #endif
      delete ksp;
    }
  };
}

// Compatibility with petsc 3.2
#if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR > 1
#define MAT_SOLVER_UMFPACK      MATSOLVERUMFPACK
#define MAT_SOLVER_MUMPS        MATSOLVERMUMPS
#define MAT_SOLVER_PASTIX       MATSOLVERPASTIX
#define MAT_SOLVER_PETSC        MATSOLVERPETSC
#define MAT_SOLVER_SPOOLES      MATSOLVERSPOOLES
#define MAT_SOLVER_SUPERLU_DIST MATSOLVERSUPERLU_DIST
#define MAT_SOLVER_SUPERLU      MATSOLVERSUPERLU
#endif

// List of available LU solvers
const std::map<std::string, const MatSolverPackage> PETScLUSolver::methods
  = boost::assign::map_list_of("default", "")
                              ("umfpack",      MAT_SOLVER_UMFPACK)
                              ("mumps",        MAT_SOLVER_MUMPS)
                              ("pastix",       MAT_SOLVER_PASTIX)
                              ("spooles",      MAT_SOLVER_SPOOLES)
                              ("superlu",      MAT_SOLVER_SUPERLU)
                              ("superlu_dist", MAT_SOLVER_SUPERLU_DIST)
                              ("petsc",        MAT_SOLVER_PETSC);

//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
PETScLUSolver::list_methods()
{
  static std::vector<std::pair<std::string, std::string> > m;

  m.push_back(std::make_pair("default",      "default LU solver"));
  m.push_back(std::make_pair("umfpack",      "UMFPACK (Unsymmetric MultiFrontal sparse LU factorization)"));
  m.push_back(std::make_pair("mumps",        "MUMPS (MUltifrontal Massively Parallel Sparse direct Solver)"));
  m.push_back(std::make_pair("pastix",       "PaStiX (Parallel Sparse matriX package)"));
  m.push_back(std::make_pair("spooles",      "SPOOLES (SParse Object Oriented Linear Equations Solver)"));
  m.push_back(std::make_pair("superlu",      "SuperLU"));
  m.push_back(std::make_pair("superlu_dist", "Parallel SuperLU"));
  m.push_back(std::make_pair("petsc",        "PETSc builtin LU solver"));

  return m;
}
//-----------------------------------------------------------------------------
Parameters PETScLUSolver::default_parameters()
{
  Parameters p(LUSolver::default_parameters());
  p.rename("petsc_lu_solver");

  // Number of threads per process for multi-threaded solvers
  p.add<uint>("num_threads");

  return p;
}
//-----------------------------------------------------------------------------
PETScLUSolver::PETScLUSolver(std::string method)
{
  // Set parameter values
  parameters = default_parameters();

  // Initialize PETSc LU solver
  init_solver(method);
}
//-----------------------------------------------------------------------------
PETScLUSolver::PETScLUSolver(boost::shared_ptr<const PETScMatrix> A,
                             std::string method) : A(A)
{
  // Check dimensions
  if (A->size(0) != A->size(1))
    error("Cannot LU factorize non-square PETSc matrix.");

  // Set parameter values
  parameters = default_parameters();

  // Initialize PETSc LU solver
  init_solver(method);
}
//-----------------------------------------------------------------------------
PETScLUSolver::~PETScLUSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void PETScLUSolver::set_operator(const boost::shared_ptr<const GenericMatrix> A)
{
  this->A = GenericTensor::down_cast<const PETScMatrix>(A);
  assert(this->A);
}
//-----------------------------------------------------------------------------
void PETScLUSolver::set_operator(const boost::shared_ptr<const PETScMatrix> A)
{
  this->A = A;
  assert(this->A);
}
//-----------------------------------------------------------------------------
const GenericMatrix& PETScLUSolver::get_operator() const
{
  if (!A)
    error("Operator for linear solver has not been set.");
  return *A;
}
//-----------------------------------------------------------------------------
dolfin::uint PETScLUSolver::solve(GenericVector& x, const GenericVector& b)
{
  assert(_ksp);
  assert(A);

  // Downcast matrix and vectors
  const PETScVector& _b = b.down_cast<PETScVector>();
  PETScVector& _x = x.down_cast<PETScVector>();

  // Check dimensions
  if (A->size(0) != b.size())
    error("Cannot LU factorize non-square PETSc matrix.");

  // Initialize solution vector if required (make compatible with A in parallel)
  if (A->size(1) != x.size())
    A->resize(x, 1);

  // Set PETSc operators (depends on factorization re-use options);
  set_petsc_operators();

  // Write a pre-solve message
  pre_report(*A);

  // FIXME: Check for solver type
  // Set number of threads if using PaStiX
  if (parameters["num_threads"].is_set())
  {
    // Use number of threads specified for LU solver
    PetscOptionsSetValue("-mat_pastix_threadnbr", parameters["num_threads"].value_str().c_str());
  }
  else
  {
    // Use global number of threads
    PetscOptionsSetValue("-mat_pastix_threadnbr", dolfin::parameters["num_threads"].value_str().c_str());
  }
  //PetscOptionsSetValue("-mat_pastix_verbose", "2");

  // Solve linear system
  KSPSolve(*_ksp, *_b.vec(), *_x.vec());

  return 1;
}
//-----------------------------------------------------------------------------
dolfin::uint PETScLUSolver::solve(const GenericMatrix& A, GenericVector& x,
                                  const GenericVector& b)
{
  return solve(A.down_cast<PETScMatrix>(), x.down_cast<PETScVector>(),
               b.down_cast<PETScVector>());
}
//-----------------------------------------------------------------------------
dolfin::uint PETScLUSolver::solve(const PETScMatrix& A, PETScVector& x,
                                  const PETScVector& b)
{
  boost::shared_ptr<const PETScMatrix> _A(&A, NoDeleter());
  set_operator(_A);
  return solve(x, b);
}
//-----------------------------------------------------------------------------
std::string PETScLUSolver::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    warning("Verbose output for PETScLUSolver not implemented, calling PETSc KSPView directly.");
    KSPView(*_ksp, PETSC_VIEWER_STDOUT_WORLD);
  }
  else
    s << "<PETScLUSolver>";

  return s.str();
}
//-----------------------------------------------------------------------------
boost::shared_ptr<KSP> PETScLUSolver::ksp() const
{
  return _ksp;
}
//-----------------------------------------------------------------------------
const MatSolverPackage PETScLUSolver::select_solver(std::string& method) const
{
  // Check package string
  if (methods.count(method) == 0)
    error("Requested PETSc LU solver '%s' is unknown,", method.c_str());

  // Choose appropriate 'default' solver
  if (method == "default")
  {
    if (MPI::num_processes() == 1)
    {
      #if PETSC_HAVE_UMFPACK
      method = "umfpack";
      #elif PETSC_HAVE_MUMPS
      method = "mumps";
      #elif PETSC_HAVE_PASTIX
      method = "pastix";
      #elif PETSC_HAVE_SUPERLU
      method = "superlu";
      #elif PETSC_HAVE_SPOOLES
      method = "spooles";
      #else
      method = "petsc";
      warning("Using PETSc native LU solver. Consider configuring PETSc with an efficient LU solver (e.g. UMFPACK, MUMPS).");
      #endif
    }
    else
    {
      #if PETSC_HAVE_MUMPS
      method = "mumps";
      #elif PETSC_HAVE_PASTIX
      method = "pastix";
      #elif PETSC_HAVE_SPOOLES
      method = "spooles";
      #elif PETSC_HAVE_SUPERLU_DIST
      method = "superlu_dist";
      #else
      error("No suitable solver for parallel LU. Consider configuring PETSc with MUMPS or SPOOLES.");
      #endif
    }
  }

  return methods.find(method)->second;
}
//-----------------------------------------------------------------------------
void PETScLUSolver::init_solver(std::string& method)
{
  // Select solver
  const MatSolverPackage solver_package = select_solver(method);

  // Destroy old solver environment if necessary
  if (_ksp)
  {
    if (!_ksp.unique())
      error("Cannot create new KSP Krylov solver. More than one object points to the underlying PETSc object.");
  }
  _ksp.reset(new KSP, PETScKSPDeleter());

  // Create solver
  if (MPI::num_processes() > 1)
    KSPCreate(PETSC_COMM_WORLD, _ksp.get());
  else
    KSPCreate(PETSC_COMM_SELF, _ksp.get());

  // Make solver preconditioner only
  KSPSetType(*_ksp, KSPPREONLY);

  // Set preconditioner to LU factorization
  PC pc;
  KSPGetPC(*_ksp, &pc);
  PCSetType(pc, PCLU);

  // Set solver package
  PCFactorSetMatSolverPackage(pc, solver_package);

  // Allow matrices with zero diagonals to be solved
  #if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR >= 1
  PCFactorSetShiftType(pc, MAT_SHIFT_NONZERO);
  PCFactorSetShiftAmount(pc, PETSC_DECIDE);
  #else
  PCFactorSetShiftNonzero(pc, PETSC_DECIDE);
  #endif
}
//-----------------------------------------------------------------------------
void PETScLUSolver::set_petsc_operators()
{
  assert(A->mat());

  // Get some parameters
  const bool reuse_fact   = parameters["reuse_factorization"];
  const bool same_pattern = parameters["same_nonzero_pattern"];

  // Set operators with appropriate preconditioner option
  if (reuse_fact)
    KSPSetOperators(*_ksp, *A->mat(), *A->mat(), SAME_PRECONDITIONER);
  else if (same_pattern)
    KSPSetOperators(*_ksp, *A->mat(), *A->mat(), SAME_NONZERO_PATTERN);
  else
    KSPSetOperators(*_ksp, *A->mat(), *A->mat(), DIFFERENT_NONZERO_PATTERN);
}
//-----------------------------------------------------------------------------
void PETScLUSolver::pre_report(const PETScMatrix& A) const
{
  const MatSolverPackage solver_type;
  PC pc;
  KSPGetPC(*_ksp, &pc);
  PCFactorGetMatSolverPackage(pc, &solver_type);

  // Get parameter
  const bool report = parameters["report"];

  if (report && dolfin::MPI::process_number() == 0)
  {
    log(PROGRESS, "Solving linear system of size %d x %d (PETSc LU solver, %s).",
        A.size(0), A.size(1), solver_type);
  }
}
//-----------------------------------------------------------------------------

#endif
