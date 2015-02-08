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

#include <dolfin/common/MPI.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/Timer.h>
#include "GenericMatrix.h"
#include "GenericVector.h"
#include "KrylovSolver.h"
#include "EigenMatrix.h"
#include "EigenPreconditioner.h"
// #include "EigenUserPreconditioner.h"
#include "EigenVector.h"
#include "EigenKrylovSolver.h"

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/../unsupported/Eigen/IterativeSolvers>

using namespace dolfin;

// Mapping from method string to description
const std::vector<std::pair<std::string, std::string> >
EigenKrylovSolver::_methods_descr =
{ {"default",    "default Krylov method"},
  {"cg",         "Conjugate gradient method"},
  {"bicgstab_ilut",   "Biconjugate gradient stabilized method (ILU)"},
  {"bicgstab",   "Biconjugate gradient stabilized method"},
  {"minres",   "Minimal residual"},
  {"gmres", "Generalised minimal residual (GMRES)"}};
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
EigenKrylovSolver::methods()
{
  return EigenKrylovSolver::_methods_descr;
}
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
EigenKrylovSolver::preconditioners()
{
  std::vector<std::pair<std::string, std::string> > pc
    = { {"default", "default"} };

  return pc;
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

  init(method);
}
//-----------------------------------------------------------------------------
EigenKrylovSolver::EigenKrylovSolver(std::string method,
                                     EigenPreconditioner& preconditioner)
  : _preconditioner(reference_to_no_delete_pointer(preconditioner))
{
  // Set parameter values
  parameters = default_parameters();

  init(method);
}
//-----------------------------------------------------------------------------
EigenKrylovSolver::EigenKrylovSolver(std::string method,
  std::shared_ptr<EigenPreconditioner> preconditioner)
  : _preconditioner(preconditioner)
{
  // Set parameter values
  parameters = default_parameters();

  init(method);
}
//-----------------------------------------------------------------------------
// EigenKrylovSolver::EigenKrylovSolver(std::string method,
//                                      EigenUserPreconditioner& preconditioner)
//   : pc_dolfin(&preconditioner),
//     preconditioner_set(false)
// {
//   // Set parameter values
//   parameters = default_parameters();

//   init(method);
// }
//-----------------------------------------------------------------------------
// EigenKrylovSolver::EigenKrylovSolver(std::string method,
//   std::shared_ptr<EigenUserPreconditioner> preconditioner)
//   : pc_dolfin(preconditioner.get()),
//     preconditioner_set(false)
// {
//   // Set parameter values
//   parameters = default_parameters();

//   init(method);
// }
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
void
EigenKrylovSolver::set_operators(std::shared_ptr<const EigenMatrix> A,
                                 std::shared_ptr<const EigenMatrix> P)
{
  _matA = A;
  _matP = P;
  dolfin_assert(_matA);
  dolfin_assert(_matP);
}
//-----------------------------------------------------------------------------
const EigenMatrix& EigenKrylovSolver::get_operator() const
{
  if (!_matA)
  {
    dolfin_error("EigenKrylovSolver.cpp",
                 "access operator for Eigen Krylov solver",
                 "Operator has not been set");
  }
  return *_matA;
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

  dolfin_assert(_matA);

  // Check dimensions
  if (_matA->size(0) != b.size())
  {
    dolfin_error("EigenKrylovSolver.cpp",
                 "unable to solve linear system with Eigen Krylov solver",
                 "Non-matching dimensions for linear system (matrix has %ld rows and right-hand side vector has %ld rows)",
                 _matA->size(0), b.size());
  }

  // Reinitialize solution vector if necessary
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
    Eigen::ConjugateGradient<eigen_matrix_type, Eigen::Whole> solver;
    num_iterations = call_solver(solver, x, b);
  }
  else if (_method == "bicgstab")
  {
    Eigen::BiCGSTAB<eigen_matrix_type> solver;
    num_iterations = call_solver(solver, x, b);
  }
  else if (_method == "bicgstab_ilut")
  {
    Eigen::BiCGSTAB<eigen_matrix_type, Eigen::IncompleteLUT<double> > solver;
    num_iterations = call_solver(solver, x, b);
  }
  else if (_method == "gmres")
  {
    Eigen::GMRES<eigen_matrix_type, Eigen::IncompleteLUT<double> > solver;
    num_iterations = call_solver(solver, x, b);
  }
  else if (_method == "minres")
  {
    Eigen::MINRES<eigen_matrix_type, Eigen::Whole> solver;
    num_iterations = call_solver(solver, x, b);
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
    s << "Eigen Krylov Solver";
  else
    s << "<EigenKrylovSolver>";

  return s.str();
}
//-----------------------------------------------------------------------------
void EigenKrylovSolver::init(const std::string& method)
{

  // Check that the requested method is known
  bool method_ok = false;
  for (auto &m : _methods_descr)
  {
    if (m.first == method)
    {
      method_ok = true;
      break;
    }
  }

  if (!method_ok)
  {
    dolfin_error("EigenKrylovSolver.cpp",
                 "create Eigen Krylov solver",
                 "Unknown Krylov method \"%s\"", method.c_str());
  }

  // Define default method
  if (method == "default")
    _method = "cg";
  else
    _method = method;
}
//-----------------------------------------------------------------------------
template <typename Solver>
std::size_t EigenKrylovSolver::call_solver(Solver& solver,
                                           GenericVector& x,
                                           const GenericVector& b)
{
  std::string timer_title = "Eigen Krylov solver (" + _method + ")";
  Timer timer(timer_title);

  const double rel_tolerance = parameters["relative_tolerance"];
  solver.setTolerance(rel_tolerance);

  const int max_iterations = parameters["maximum_iterations"];
  solver.setMaxIterations(max_iterations);

  solver.compute(_matA->mat());
  if (solver.info() != Eigen::Success)
  {
    dolfin_error("EigenKrylovSolver.cpp",
                 "prepare Krylov solver",
                 "Preconditioner might fail");
  }

  EigenVector& _x = as_type<EigenVector>(x);
  const EigenVector& _b = as_type<const EigenVector>(b);

  const bool nonzero_guess = parameters["nonzero_initial_guess"];
  if (nonzero_guess)
    _x.vec() = solver.solveWithGuess(_b.vec(), _x.vec());
  else
    _x.vec() = solver.solve(_b.vec());
  const int num_iterations = solver.iterations();

  if (solver.info() != Eigen::Success)
  {
    if (num_iterations >= max_iterations)
    {
      dolfin_error("EigenKrylovSolver.cpp",
                   "solve A.x = b",
                   "Max iterations (%d) exceeded", max_iterations);
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
