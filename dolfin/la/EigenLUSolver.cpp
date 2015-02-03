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
// First added:  2015-02-03

#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/Timer.h>
#include <dolfin/parameter/GlobalParameters.h>

#include <Eigen/SparseLU>

#include "LUSolver.h"
#include "EigenMatrix.h"
#include "EigenVector.h"
#include "EigenLUSolver.h"

using namespace dolfin;

// List of available LU solvers
const std::map<std::string, std::string> EigenLUSolver::_methods
= { {"default", ""},
    {"SparseLU", "SparseLU"}};
//-----------------------------------------------------------------------------
const std::vector<std::pair<std::string, std::string> >
EigenLUSolver::_methods_descr
= { {"default", "default LU solver"},
    {"SparseLU", "Supernodal LU factorization for general matrices"}};
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
EigenLUSolver::methods()
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

  // Initialize Eigen LU solver
  init_solver(method);
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

  // Initialize Eigen LU solver
  init_solver(method);
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
  return solve(x, b, false);
}
//-----------------------------------------------------------------------------
std::size_t EigenLUSolver::solve(GenericVector& x, const GenericVector& b,
                                 bool transpose)
{
  Timer timer("Eigen LU solver");

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

  // Initialize solution vector if required (make compatible with A in
  // parallel)
  if (x.empty())
    _matA->init_vector(x, 1);

  // Copy to column major format
  Eigen::SparseMatrix<double, Eigen::ColMajor> _A;
  if (transpose)
    _A = _matA->mat().transpose();
  else
    _A = _matA->mat();

  _A.makeCompressed();

  Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::ColMajor>,
                  Eigen::COLAMDOrdering<int> > solver;

  solver.analyzePattern(_A);
  solver.factorize(_A);

  // Solve linear system
  _x.vec() = solver.solve(_b.vec());

  return 1;
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
std::size_t EigenLUSolver::solve_transpose(GenericVector& x,
                                           const GenericVector& b)
{
return solve(x, b, true);
}
//-----------------------------------------------------------------------------
std::size_t EigenLUSolver::solve_transpose(const GenericLinearOperator& A,
                                           GenericVector& x,
                                           const GenericVector& b)
{
  return solve_transpose(as_type<const EigenMatrix>(require_matrix(A)),
                         as_type<EigenVector>(x),
                         as_type<const EigenVector>(b));
}
//-----------------------------------------------------------------------------
std::size_t EigenLUSolver::solve_transpose(const EigenMatrix& A,
                                           EigenVector& x,
                                           const EigenVector& b)
{
  std::shared_ptr<const EigenMatrix> _matA(&A, NoDeleter());
  set_operator(_matA);
  return solve_transpose(x, b);
}
//-----------------------------------------------------------------------------
std::string EigenLUSolver::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << "Eigen LUSolver\n";
  }
  else
    s << "<EigenLUSolver>";

  return s.str();
}
//-----------------------------------------------------------------------------
const std::string EigenLUSolver::select_solver(std::string& method) const
{
  // Check package string
  if (_methods.count(method) == 0)
  {
    dolfin_error("EigenLUSolver.cpp",
                 "solve linear system using Eigen LU solver",
                 "Unknown LU method \"%s\"", method.c_str());
  }

  // Choose appropriate 'default' solver
  if (method == "default")
    method = "SparseLU";

  return _methods.find(method)->second;
}
//-----------------------------------------------------------------------------
void EigenLUSolver::init_solver(std::string& method)
{
  // Do nothing for now
}
//-----------------------------------------------------------------------------
