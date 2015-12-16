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

#ifdef HAS_TRILINOS

#include <dolfin/common/constants.h>
#include <dolfin/common/Timer.h>
#include <dolfin/log/log.h>
#include <dolfin/common/MPI.h>
#include <dolfin/parameter/GlobalParameters.h>

#include "LUSolver.h"
#include "TpetraMatrix.h"
#include "TpetraVector.h"
#include "Amesos2LUSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::map<std::string, std::string>
Amesos2LUSolver::methods()
{
  std::map<std::string, std::string> amesos2_methods;
  amesos2_methods["default"] = "default method";
#ifdef HAVE_AMESOS2_KLU2
  amesos2_methods["KLU2"] = "Built in KLU2";
#endif
#ifdef HAVE_AMESOS2_BASKER
  amesos2_methods["Basker"] = "Basker";
#endif
#ifdef HAVE_AMESOS2_SUPERLU
  amesos2_methods["Superlu"] = "Superlu";
#endif
#ifdef HAVE_AMESOS2_LAPACK
  amesos2_methods["Lapack"] = "Lapack (for testing)";
#endif
#ifdef HAVE_AMESOS2_MUMPS
  amesos2_methods["Mumps"] = "MUMPS";
#endif
#ifdef HAVE_AMESOS2_SUPERLUDIST
  amesos2_methods["Superludist"] = "Superludist";
#endif
  return amesos2_methods;
}
//-----------------------------------------------------------------------------
Parameters Amesos2LUSolver::default_parameters()
{
  Parameters p(LUSolver::default_parameters());
  p.rename("amesos2_lu_solver");

  return p;
}
//-----------------------------------------------------------------------------
Amesos2LUSolver::Amesos2LUSolver(std::string method)
{
  // Set parameter values
  parameters = default_parameters();

  // Initialize Tpetra LU solver
  init_solver(method);
}
//-----------------------------------------------------------------------------
Amesos2LUSolver::Amesos2LUSolver(std::shared_ptr<const TpetraMatrix> A,
                                 std::string method) : _matA(A)
{
  // Check dimensions
  if (A->size(0) != A->size(1))
  {
    dolfin_error("Amesos2LUSolver.cpp",
                 "create Amesos2 LU solver",
                 "Cannot LU factorize non-square TpetraMatrix");
  }

  // Set parameter values
  parameters = default_parameters();

  // Initialize Amesos2 LU solver
  init_solver(method);
}
//-----------------------------------------------------------------------------
Amesos2LUSolver::~Amesos2LUSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void
Amesos2LUSolver::set_operator(std::shared_ptr<const GenericLinearOperator> A)
{
  // Attempt to cast as TpetraMatrix
  std::shared_ptr<const TpetraMatrix> mat
    = as_type<const TpetraMatrix>(A);
  dolfin_assert(mat);

  // Set operator
  set_operator(mat);
}
//-----------------------------------------------------------------------------
void Amesos2LUSolver::set_operator(std::shared_ptr<const TpetraMatrix> A)
{
  _matA = A;
  dolfin_assert(_matA);

  if (_matA->mat().is_null())
  {
    dolfin_error("Amesos2LUSolver.cpp",
                 "set operator (Amesos2LUSolver::set_operator)",
                 "cannot set operator if matrix has not been initialized");
  }

  _solver->setA(_matA->mat());
  // ierr = KSPSetOperators(_ksp, _matA->mat(), _matA->mat());

}
//-----------------------------------------------------------------------------
const GenericLinearOperator& Amesos2LUSolver::get_operator() const
{
  if (!_matA)
  {
    dolfin_error("Amesos2LUSolver.cpp",
                 "access operator of Tpetra LU solver",
                 "Operator has not been set");
  }
  return *_matA;
}
//-----------------------------------------------------------------------------
std::size_t Amesos2LUSolver::solve(GenericVector& x, const GenericVector& b)
{
  Timer timer("Amesos2 LU solver");

  dolfin_assert(_matA);
  dolfin_assert(!_matA->mat().is_null());

  // Downcast matrix and vectors
  const TpetraVector& _b = as_type<const TpetraVector>(b);
  TpetraVector& _x = as_type<TpetraVector>(x);

  // Check dimensions
  if (_matA->size(0) != b.size())
  {
    dolfin_error("Amesos2LUSolver.cpp",
                 "solve linear system using Amesos2 LU solver",
                 "Cannot factorize non-square TpetraMatrix");
  }

  // Initialize solution vector if required
  // (make compatible with A in parallel)
  if (x.empty())
    _matA->init_vector(x, 1);

  if (Amesos2::query(_method_name) == false)
  {
    dolfin_error("Amesos2LUSolver.cpp",
                 "set solver method",
                 "\"%s\" not supported", _method_name.c_str());
  }

  if (_solver.is_null())
  {
    _solver = Amesos2::create(_method_name, _matA->mat(),
                              _x.vec(),
      Teuchos::rcp_dynamic_cast<const TpetraVector::vector_type>(_b.vec()));
  }
  else
  {
    _solver->setX(_x.vec());
    _solver->setB(Teuchos::rcp_dynamic_cast<const TpetraVector::vector_type>(_b.vec()));
  }
  _solver->solve();

  return 1;
}
//-----------------------------------------------------------------------------
std::size_t Amesos2LUSolver::solve(const GenericLinearOperator& A,
                                 GenericVector& x,
                                 const GenericVector& b)
{
  return solve(as_type<const TpetraMatrix>(A),
               as_type<TpetraVector>(x),
               as_type<const TpetraVector>(b));
}
//-----------------------------------------------------------------------------
std::size_t Amesos2LUSolver::solve(const TpetraMatrix& A,
                                   TpetraVector& x,
                                   const TpetraVector& b)
{
  _matA = std::make_shared<TpetraMatrix>(A);
  return solve(x, b);
}
//-----------------------------------------------------------------------------
std::string Amesos2LUSolver::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    warning("Verbose output for Amesos2LUSolver not implemented yet.");
  }
  else
    s << "<Amesos2LUSolver>";

  return s.str();
}
//-----------------------------------------------------------------------------
void Amesos2LUSolver::init_solver(std::string& method)
{
  _method_name = method;
  if (method == "default")
  {
    #ifdef HAVE_AMESOS2_KLU2
    _method_name = "KLU2";
    #else
    dolfin_error("Amesos2LUSolver.cpp", "initialise solver",
      "Default KLU2 solver not found");
    #endif
}

}
//-----------------------------------------------------------------------------
#endif
