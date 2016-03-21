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
//
// First added:  2015-02-04

#include <iostream> // Seem to be missing some Eigen headers
#include <map>
#include <string>

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/../unsupported/Eigen/IterativeSolvers>

#include <dolfin/common/types.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/Timer.h>
#include <dolfin/log/log.h>
#include "EigenMatrix.h"
#include "EigenVector.h"
#include "GenericMatrix.h"
#include "GenericVector.h"
#include "KrylovSolver.h"
#include "EigenKrylovSolver.h"

using namespace dolfin;

// Mapping from method string to description
const std::map<std::string, std::string>
EigenKrylovSolver::_methods_descr
= { {"default",  "default Eigen Krylov method"},
    {"cg",       "Conjugate gradient method"},
    {"bicgstab", "Biconjugate gradient stabilized method"},
    {"minres",   "Minimal residual"},
    {"gmres",    "Generalised minimal residual (GMRES)"}};

// Mapping from preconditioner string to description
const std::map<std::string, std::string>
EigenKrylovSolver::_pcs_descr
= { {"default", "default"},
    {"none",    "None"},
    {"jacobi",  "Jacobi"},
    {"ilu",     "Incomplete LU"} };
//-----------------------------------------------------------------------------
std::map<std::string, std::string> EigenKrylovSolver::methods()
{
  return EigenKrylovSolver::_methods_descr;
}
//-----------------------------------------------------------------------------
std::map<std::string, std::string> EigenKrylovSolver::preconditioners()
{
  return EigenKrylovSolver::_pcs_descr;
}
//-----------------------------------------------------------------------------
Parameters EigenKrylovSolver::default_parameters()
{
  Parameters p(KrylovSolver::default_parameters());
  p.rename("eigen_krylov_solver");
  return p;
}
//-----------------------------------------------------------------------------
EigenKrylovSolver::EigenKrylovSolver(std::string method,
                                     std::string preconditioner)
{
  // Set parameter values
  parameters = default_parameters();

  // Initialise
  init(method, preconditioner);
}
//-----------------------------------------------------------------------------
EigenKrylovSolver::~EigenKrylovSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void
EigenKrylovSolver::set_operator(std::shared_ptr<const GenericLinearOperator> A)
{
  set_operators(A, A);
}
//-----------------------------------------------------------------------------
void EigenKrylovSolver::set_operator(std::shared_ptr<const EigenMatrix> A)
{
  set_operators(A, A);
}
//-----------------------------------------------------------------------------
void EigenKrylovSolver::set_operators(
  std::shared_ptr<const  GenericLinearOperator> A,
  std::shared_ptr<const GenericLinearOperator> P)
{
  set_operators(as_type<const EigenMatrix>(A),
                as_type<const EigenMatrix>(P));
}
//-----------------------------------------------------------------------------
void EigenKrylovSolver::set_operators(std::shared_ptr<const EigenMatrix> A,
                                      std::shared_ptr<const EigenMatrix> P)
{
  _matA = A;
  _matP = P;
  dolfin_assert(_matA);
  dolfin_assert(_matP);
}
//-----------------------------------------------------------------------------
std::shared_ptr<const EigenMatrix> EigenKrylovSolver::get_operator() const
{
  if (!_matA)
  {
    dolfin_error("EigenKrylovSolver.cpp",
                 "access operator for Eigen Krylov solver",
                 "Operator has not been set");
  }
  return _matA;
}
//-----------------------------------------------------------------------------
std::size_t EigenKrylovSolver::solve(GenericVector& x, const GenericVector& b)
{
  return solve(as_type<EigenVector>(x), as_type<const EigenVector>(b));
}
//-----------------------------------------------------------------------------
std::size_t EigenKrylovSolver::solve(const GenericLinearOperator& A,
                                     GenericVector& x,
                                     const GenericVector& b)
{
  return solve(as_type<const EigenMatrix>(A), as_type<EigenVector>(x),
               as_type<const EigenVector>(b));
}
//-----------------------------------------------------------------------------
std::size_t EigenKrylovSolver::solve(EigenVector& x, const EigenVector& b)
{
  Timer timer("Eigen Krylov solver");

  // Check dimensions
  dolfin_assert(_matA);
  if (_matA->size(0) != b.size())
  {
    dolfin_error("EigenKrylovSolver.cpp",
                 "unable to solve linear system with Eigen Krylov solver",
                 "Non-matching dimensions for linear system (matrix has %ld rows and right-hand side vector has %ld rows)",
                 _matA->size(0), b.size());
  }

  // Re-initialize solution vector if necessary
  if (x.empty())
  {
    _matA->init_vector(x, 1);
    x.zero();
  }

  log(PROGRESS, "Eigen Krylov solver starting to solve %i x %i system.",
      _matA->size(0), _matA->size(1));

  std::size_t num_iterations = 0;

  if (_method == "cg")
  {
    if (_pc == "none")
    {
      Eigen::ConjugateGradient<EigenMatrix::eigen_matrix_type,
                               Eigen::Upper|Eigen::Lower,
                               Eigen::IdentityPreconditioner> solver;
      num_iterations = call_solver(solver, x, b);
    }
    else if (_pc == "jacobi")
    {
      Eigen::ConjugateGradient<EigenMatrix::eigen_matrix_type,
                               Eigen::Upper|Eigen::Lower,
                               Eigen::DiagonalPreconditioner<double>> solver;
      num_iterations = call_solver(solver, x, b);
    }
    else if (_pc == "ilu")
    {
      Eigen::ConjugateGradient<EigenMatrix::eigen_matrix_type,
                               Eigen::Upper|Eigen::Lower,
                               Eigen::IncompleteLUT<double>> solver;
      num_iterations = call_solver(solver, x, b);
    }
    else
    {
      Eigen::ConjugateGradient<EigenMatrix::eigen_matrix_type,
                               Eigen::Upper|Eigen::Lower> solver;
      num_iterations = call_solver(solver, x, b);
    }
  }
  else if (_method == "bicgstab")
  {
    if (_pc == "none")
    {
      Eigen::BiCGSTAB<EigenMatrix::eigen_matrix_type,
                      Eigen::IdentityPreconditioner> solver;
      num_iterations = call_solver(solver, x, b);
    }
    else if (_pc == "jacobi")
    {
      Eigen::BiCGSTAB<EigenMatrix::eigen_matrix_type,
                      Eigen::DiagonalPreconditioner<double>> solver;
      num_iterations = call_solver(solver, x, b);
    }
    else if (_pc == "ilu")
    {
      Eigen::BiCGSTAB<EigenMatrix::eigen_matrix_type,
                      Eigen::IncompleteLUT<double>> solver;
      num_iterations = call_solver(solver, x, b);
    }
    else
    {
      Eigen::BiCGSTAB<EigenMatrix::eigen_matrix_type> solver;
      num_iterations = call_solver(solver, x, b);
    }
  }
  else if (_method == "gmres")
  {
    if (_pc == "none")
    {
      Eigen::GMRES<EigenMatrix::eigen_matrix_type,
                   Eigen::IdentityPreconditioner> solver;
      num_iterations = call_solver(solver, x, b);
    }
    else if (_pc == "jacobi")
    {
      Eigen::GMRES<EigenMatrix::eigen_matrix_type,
                   Eigen::DiagonalPreconditioner<double>> solver;
      num_iterations = call_solver(solver, x, b);
    }
    else if (_pc == "ilu")
    {
      Eigen::GMRES<EigenMatrix::eigen_matrix_type,
                   Eigen::IncompleteLUT<double>> solver;
      num_iterations = call_solver(solver, x, b);
    }
    else
    {
      Eigen::GMRES<EigenMatrix::eigen_matrix_type> solver;
      num_iterations = call_solver(solver, x, b);
    }
  }
  else if (_method == "minres")
  {
    if (_pc == "none")
    {
      Eigen::MINRES<EigenMatrix::eigen_matrix_type, Eigen::Upper|Eigen::Lower,
                    Eigen::IdentityPreconditioner> solver;
      num_iterations = call_solver(solver, x, b);
    }
    else if (_pc == "jacobi")
    {
      Eigen::MINRES<EigenMatrix::eigen_matrix_type, Eigen::Upper|Eigen::Lower,
                    Eigen::DiagonalPreconditioner<double>> solver;
      num_iterations = call_solver(solver, x, b);
    }
    else if (_pc == "ilu")
    {
      Eigen::MINRES<EigenMatrix::eigen_matrix_type, Eigen::Upper|Eigen::Lower,
                    Eigen::IncompleteLUT<double>> solver;
      num_iterations = call_solver(solver, x, b);
    }
    else
    {
      Eigen::MINRES<EigenMatrix::eigen_matrix_type> solver;
      num_iterations = call_solver(solver, x, b);
    }
  }

  return num_iterations;
}
//-----------------------------------------------------------------------------
std::size_t EigenKrylovSolver::solve(const EigenMatrix& A, EigenVector& x,
                                     const EigenVector& b)
{
  // Set operator
  std::shared_ptr<const EigenMatrix> Atmp(&A, NoDeleter());
  set_operator(Atmp);

  // Call solve
  return solve(x, b);
}
//-----------------------------------------------------------------------------
std::string EigenKrylovSolver::str(bool verbose) const
{
  std::stringstream s;
  if (verbose)
    s << "Eigen Krylov Solver (" << _method << ", "
      << _pc << ")" << std::endl;
  else
    s << "<EigenKrylovSolver>";

  return s.str();
}
//-----------------------------------------------------------------------------
void EigenKrylovSolver::init(const std::string method,
                             const std::string pc)
{
  // Check that the requested solver method is known
  if (_methods_descr.find(method) == _methods_descr.end())
  {
    dolfin_error("EigenKrylovSolver.cpp",
                 "create Eigen Krylov solver",
                 "Unknown Krylov method \"%s\"", method.c_str());
  }

  // Check that the requested preconditioner is known
  if (_pcs_descr.find(pc) == _pcs_descr.end())
  {
    dolfin_error("EigenKrylovSolver.cpp",
                 "create Eigen Krylov solver",
                 "Unknown preconditioner \"%s\"", pc.c_str());
  }

  // Set method and preconditioner
  _method = (method == "default" ? "gmres" : method);
  _pc = pc;
}
//-----------------------------------------------------------------------------
template <typename Solver>
std::size_t EigenKrylovSolver::call_solver(Solver& solver,
                                           GenericVector& x,
                                           const GenericVector& b)
{
  std::string timer_title = "Eigen Krylov solver (" + _method + ")";
  Timer timer(timer_title);

  EigenVector& _x = as_type<EigenVector>(x);
  const EigenVector& _b = as_type<const EigenVector>(b);

  if (parameters["relative_tolerance"].is_set())
    solver.setTolerance(parameters["relative_tolerance"]);

  if (parameters["maximum_iterations"].is_set())
    solver.setMaxIterations(parameters["maximum_iterations"]);

  // Prepare solver
  solver.compute(_matA->mat());
  if (solver.info() != Eigen::Success)
  {
    dolfin_error("EigenKrylovSolver.cpp",
                 "prepare Krylov solver",
                 "Preconditioner might fail");
  }

  // Call approriate solve function
  if (parameters["nonzero_initial_guess"].is_set())
  {
    const bool nonzero_guess = parameters["nonzero_initial_guess"];
    if (nonzero_guess)
      _x.vec() = solver.solveWithGuess(_b.vec(), _x.vec());
    else
      _x.vec() = solver.solve(_b.vec());
  }
  else
    _x.vec() = solver.solve(_b.vec());

  // Get number of solver iterations
  const int num_iterations = solver.iterations();

  // Handle case that solver fails to converge
  bool error_on_nonconvergence = parameters["error_on_nonconvergence"].is_set() ? parameters["error_on_nonconvergence"].is_set() : false;
  if (solver.info() != Eigen::Success)
  {
    if (num_iterations >= solver.maxIterations())
    {
      if (error_on_nonconvergence)
      {
        dolfin_error("EigenKrylovSolver.cpp",
                     "solve A.x = b",
                     "Max iterations (%d) exceeded", solver.maxIterations());
      }
      else
      {
        warning("Krylov solver did not converge in %i iterations",
                solver.maxIterations());
      }
    }
    else
    {
      dolfin_error("EigenKrylovSolver.cpp",
                   "solve A.x = b",
                   "Solver failed");
    }
  }

  return num_iterations;
}
//-----------------------------------------------------------------------------
