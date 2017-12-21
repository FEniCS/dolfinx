// Copyright (C) 2015 Chris Richardson
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

#include <dolfin/common/types.h>
#include <Eigen/SparseLU>
#ifdef HAS_CHOLMOD
// Works with Cholmod downloaded by PETSc
// Eigen uses deprecated UF_long
// now redefined as SuiteSparse_long
#ifndef UF_long
#define UF_long     SuiteSparse_long
#define UF_long_max SuiteSparse_long_max
#define UF_long_idd SuiteSparse_long_idd
#define UF_long_id  SuiteSparse_long_id
#endif
#include <Eigen/CholmodSupport>
#endif
#ifdef HAS_UMFPACK
// Works with Suitesparse downloaded by PETSc
#include <Eigen/UmfPackSupport>
#endif
#ifdef EIGEN_PARDISO_SUPPORT
// Requires Intel MKL
#include <Eigen/PardisoSupport>
#endif
#ifdef EIGEN_SUPERLU_SUPPORT
// SuperLU bundled with PETSc is not compatible
#include <Eigen/SuperLUSupport>
#endif
#ifdef EIGEN_PASTIX_SUPPORT
// Seems to require COMPLEX support
#include <Eigen/PaStiXSupport>
#endif

#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/Timer.h>
#include <dolfin/parameter/GlobalParameters.h>
#include "EigenMatrix.h"
#include "EigenVector.h"
#include "LUSolver.h"
#include "EigenLUSolver.h"

using namespace dolfin;

// List of available LU solvers
const std::map<std::string, std::string>
EigenLUSolver::_methods_descr
= { {"default", "default LU solver"},
    {"sparselu", "Supernodal LU factorization for general matrices"},
    {"cholesky", "Simplicial LDLT"},
#ifdef HAS_CHOLMOD
    {"cholmod", "'CHOLMOD' sparse Cholesky factorisation"},
#endif
#ifdef HAS_UMFPACK
    {"umfpack", "UMFPACK (Unsymmetric MultiFrontal sparse LU factorization)"},
#endif
#ifdef EIGEN_PARDISO_SUPPORT
    {"pardiso", "Intel MKL Pardiso"},
#endif
#ifdef EIGEN_SUPERLU_SUPPORT
    {"superlu", "SuperLU"},
#endif
#ifdef EIGEN_PASTIX_SUPPORT
    {"pastix", "PaStiX (Parallel Sparse matriX package)"}
#endif
};
//-----------------------------------------------------------------------------
std::map<std::string, std::string> EigenLUSolver::methods()
{
  return EigenLUSolver::_methods_descr;
}
//-----------------------------------------------------------------------------
Parameters EigenLUSolver::default_parameters()
{
  Parameters p(LUSolver::default_parameters());
  p.rename("eigen_lu_solver");
  return p;
}
//-----------------------------------------------------------------------------
EigenLUSolver::EigenLUSolver(std::string method)
{
  // Set parameter values
  parameters = default_parameters();

  _method = select_solver(method);
}
//-----------------------------------------------------------------------------
EigenLUSolver::EigenLUSolver(std::shared_ptr<const EigenMatrix> A,
                             std::string method) : _matA(A)
{
  // Check dimensions
  if (A->size(0) != A->size(1))
  {
    dolfin_error("EigenLUSolver.cpp",
                 "create Eigen LU solver",
                 "Cannot LU factorize non-square EigenMatrix");
  }

  // Set parameter values
  parameters = default_parameters();

  _method = select_solver(method);
}
//-----------------------------------------------------------------------------
EigenLUSolver::~EigenLUSolver()
{
}
//-----------------------------------------------------------------------------
void
EigenLUSolver::set_operator(std::shared_ptr<const GenericLinearOperator> A)
{
  // Attempt to cast as EigenMatrix
  std::shared_ptr<const EigenMatrix> mat
    = as_type<const EigenMatrix>(require_matrix(A));
  dolfin_assert(mat);

  // Set operator
  set_operator(mat);
}
//-----------------------------------------------------------------------------
void EigenLUSolver::set_operator(std::shared_ptr<const EigenMatrix> A)
{
  _matA = A;
  dolfin_assert(_matA);
  dolfin_assert(!_matA->empty());
}
//-----------------------------------------------------------------------------
const GenericLinearOperator& EigenLUSolver::get_operator() const
{
  if (!_matA)
  {
    dolfin_error("EigenLUSolver.cpp",
                 "access operator of Eigen LU solver",
                 "Operator has not been set");
  }
  return *_matA;
}
//-----------------------------------------------------------------------------
std::size_t EigenLUSolver::solve(GenericVector& x, const GenericVector& b)
{
  if (_method == "sparselu")
  {
    Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::ColMajor>,
                    Eigen::COLAMDOrdering<int>> solver;
    call_solver(solver, x, b);
  }
  else if (_method == "cholesky")
  {
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double, Eigen::ColMajor>,
                          Eigen::Lower> solver;
    call_solver(solver, x, b);
  }
#ifdef HAS_CHOLMOD
  else if (_method == "cholmod")
  {
    Eigen::CholmodDecomposition<Eigen::SparseMatrix<double, Eigen::ColMajor>,
                                Eigen::Lower> solver;
    solver.setMode(Eigen::CholmodLDLt);
    call_solver(solver, x, b);
  }
#endif
#ifdef EIGEN_PASTIX_SUPPORT
  else if (_method == "pastix")
  {
    Eigen::PastixLU<Eigen::SparseMatrix<double, Eigen::ColMajor>> solver;
    call_solver(solver, x, b);
  }
#endif
#ifdef EIGEN_PARDISO_SUPPORT
  else if (_method == "pardiso")
  {
    Eigen::PardisoLU<Eigen::SparseMatrix<double, Eigen::ColMajor>> solver;
    call_solver(solver, x, b);
  }
#endif
#ifdef EIGEN_SUPERLU_SUPPORT
  else if (_method == "superlu")
  {
    Eigen::SuperLU<Eigen::SparseMatrix<double, Eigen::ColMajor>> solver;
    call_solver(solver, x, b);
  }
#endif
#ifdef HAS_UMFPACK
  else if (_method == "umfpack")
  {
    Eigen::UmfPackLU<Eigen::SparseMatrix<double, Eigen::ColMajor>> solver;
    call_solver(solver, x, b);
  }
#endif
  else
    dolfin_error("EigenLUSolver.cpp", "solve A.x =b",
                 "Unknown method \"%s\"", _method.c_str());

  return 1;
}
//-----------------------------------------------------------------------------
template <typename Solver>
void EigenLUSolver::call_solver(Solver& solver, GenericVector& x,
                                const GenericVector& b)
{
  const std::string timer_title = "Eigen LU solver (" + _method + ")";
  Timer timer(timer_title);

  dolfin_assert(_matA);

  // Downcast matrix and vectors
  const EigenVector& _b = as_type<const EigenVector>(b);
  EigenVector& _x = as_type<EigenVector>(x);

  // Check dimensions
  if (_matA->size(0) != b.size())
  {
    dolfin_error("EigenLUSolver.cpp",
                 "solve linear system using Eigen LU solver",
                 "Cannot factorize non-square Eigen matrix");
  }

  // Initialize solution vector if required
  if (x.empty())
    _matA->init_vector(x, 1);

  // Copy to format suitable for solver
  // Eigen wants ColMajor matrices for solver
  // FIXME: Do we want this? It could affect re-assembly performance
  typename Solver::MatrixType _A;
  _A = _matA->mat();

  // Compress matrix
  // Most solvers require a compressed matrix
  _A.makeCompressed();

  // Factorize matrix
  solver.compute(_A);

  if (solver.info() != Eigen::Success)
  {
    dolfin_error("EigenLUSolver.cpp",
                 "compute matrix factorisation",
                 "The provided data did not satisfy the prerequisites");
  }

  // Solve linear system
  dolfin_assert(_b.vec());
  dolfin_assert(_x.vec());
  *(_x.vec()) = solver.solve(*(_b.vec()));
  if (solver.info() != Eigen::Success)
  {
    dolfin_error("EigenLUSolver.cpp",
                 "solve A.x = b",
                 "Solver failed");
  }
}
//-----------------------------------------------------------------------------
std::size_t EigenLUSolver::solve(const GenericLinearOperator& A,
                                 GenericVector& x,
                                 const GenericVector& b)
{
  return solve(as_type<const EigenMatrix>(require_matrix(A)),
               as_type<EigenVector>(x),
               as_type<const EigenVector>(b));
}
//-----------------------------------------------------------------------------
std::size_t EigenLUSolver::solve(const EigenMatrix& A, EigenVector& x,
                                 const EigenVector& b)
{
  std::shared_ptr<const EigenMatrix> Atmp(&A, NoDeleter());
  set_operator(Atmp);
  return solve(x, b);
}
//-----------------------------------------------------------------------------
std::string EigenLUSolver::str(bool verbose) const
{
  std::stringstream s;
  if (verbose)
    s << "Eigen LUSolver (" << _method << ")" << std::endl;
  else
    s << "<EigenLUSolver>";

  return s.str();
}
//-----------------------------------------------------------------------------
std::string EigenLUSolver::select_solver(const std::string method) const
{
  if (method == "default")
    return "sparselu";

  if (_methods_descr.find(method) == _methods_descr.end())
  {
    dolfin_error("EigenLUSolver.cpp",
                 "solve linear system using Eigen LU solver",
                 "Unknown LU method \"%s\"", method.c_str());
  }

  return method;
}
//-----------------------------------------------------------------------------
