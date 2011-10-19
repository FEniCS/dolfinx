// Copyright (C) 2007-2011 Anders Logg
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
// Modified by Ola Skavhaug 2008.
// Modified by Garth N. Wells 2011.
//
// First added:  2007-04-30
// Last changed: 2011-10-07

#include <boost/scoped_ptr.hpp>

#include <dolfin/common/Timer.h>
#include <dolfin/log/Table.h>
#include <dolfin/log/LogStream.h>
#include "GenericMatrix.h"
#include "GenericVector.h"
#include "LinearAlgebraFactory.h"
#include "DefaultFactory.h"
#include "LinearSolver.h"
#include "solve.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::uint dolfin::solve(const GenericMatrix& A,
                           GenericVector& x,
                           const GenericVector& b,
                           std::string method,
                           std::string preconditioner)
{
  Timer timer("Solving linear system");
  LinearSolver solver(method, preconditioner);
  return solver.solve(A, x, b);
}
//-----------------------------------------------------------------------------
void dolfin::list_linear_solver_methods()
{
  // Get methods
  std::vector<std::pair<std::string, std::string> >
    methods = linear_solver_methods();

  // Pretty-print list of methods
  Table t("Solver method", false);
  for (uint i = 0; i < methods.size(); i++)
    t(methods[i].first, "Description") = methods[i].second;
  cout << t.str(true) << endl;
}
//-----------------------------------------------------------------------------
void dolfin::list_lu_solver_methods()
{
  // Get methods
  std::vector<std::pair<std::string, std::string> >
    methods = lu_solver_methods();

  // Pretty-print list of methods
  Table t("LU method", false);
  for (uint i = 0; i < methods.size(); i++)
    t(methods[i].first, "Description") = methods[i].second;
  cout << t.str(true) << endl;
}
//-----------------------------------------------------------------------------
void dolfin::list_krylov_solver_methods()
{
  // Get methods
  std::vector<std::pair<std::string, std::string> >
    methods = krylov_solver_methods();

  // Pretty-print list of methods
  Table t("Krylov method", false);
  for (uint i = 0; i < methods.size(); i++)
    t(methods[i].first, "Description") = methods[i].second;
  cout << t.str(true) << endl;
}
//-----------------------------------------------------------------------------
void dolfin::list_krylov_solver_preconditioners()
{
  // Get preconditioners
  std::vector<std::pair<std::string, std::string> >
    preconditioners = krylov_solver_preconditioners();

  // Pretty-print list of preconditioners
  Table t("Preconditioner", false);
  for (uint i = 0; i < preconditioners.size(); i++)
    t(preconditioners[i].first, "Description") = preconditioners[i].second;
  cout << t.str(true) << endl;
}
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
dolfin::linear_solver_methods()
{
  std::vector<std::pair<std::string, std::string> >
    lu_methods = DefaultFactory::factory().lu_solver_methods();
  std::vector<std::pair<std::string, std::string> >
    krylov_methods = DefaultFactory::factory().krylov_solver_methods();

  std::vector<std::pair<std::string, std::string> >
    methods;
  for (uint i = 0; i < lu_methods.size(); i++)
    methods.push_back(lu_methods[i]);
  for (uint i = 0; i < krylov_methods.size(); i++)
    methods.push_back(krylov_methods[i]);

  return methods;
}
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> > dolfin::lu_solver_methods()
{
  return DefaultFactory::factory().lu_solver_methods();
}
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
dolfin::krylov_solver_methods()
{
  return DefaultFactory::factory().krylov_solver_methods();
}
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
dolfin::krylov_solver_preconditioners()
{
  return DefaultFactory::factory().krylov_solver_preconditioners();
}
//-----------------------------------------------------------------------------
double dolfin::residual(const GenericMatrix& A,
                        const GenericVector& x,
                        const GenericVector& b)
{
  boost::scoped_ptr<GenericVector> y(A.factory().create_vector());
  A.mult(x, *y);
  *y -= b;
  return y->norm("l2");
}
//-----------------------------------------------------------------------------
double dolfin::normalize(GenericVector& x, std::string normalization_type)
{
  double c = 0.0;
  if (normalization_type == "l2")
  {
    c = x.norm("l2");
    x /= c;
  }
  else if (normalization_type == "average")
  {
    boost::scoped_ptr<GenericVector> y(x.factory().create_vector());
    y->resize(x.size());
    (*y) = 1.0 / static_cast<double>(x.size());
    c = x.inner(*y);
    (*y) = c;
    x -= (*y);
  }
  else
    error("Unknown normalization type.");

  return c;
}
//-----------------------------------------------------------------------------

