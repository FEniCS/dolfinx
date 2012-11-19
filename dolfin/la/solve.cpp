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
// Modified by Ola Skavhaug 2008
// Modified by Garth N. Wells 2011
// Modified by Mikael Mortensen 2011
//
// First added:  2007-04-30
// Last changed: 2011-12-21

#include <boost/shared_ptr.hpp>
#include <boost/assign/list_of.hpp>

#include <dolfin/common/Timer.h>
#include <dolfin/log/Table.h>
#include <dolfin/log/LogStream.h>
#include "GenericLinearOperator.h"
#include "GenericVector.h"
#include "GenericLinearAlgebraFactory.h"
#include "DefaultFactory.h"
#include "LinearSolver.h"
#include "solve.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
unsigned int dolfin::solve(const GenericLinearOperator& A,
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
  for (unsigned int i = 0; i < methods.size(); i++)
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
  for (unsigned int i = 0; i < methods.size(); i++)
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
  for (unsigned int i = 0; i < methods.size(); i++)
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
  for (unsigned int i = 0; i < preconditioners.size(); i++)
    t(preconditioners[i].first, "Description") = preconditioners[i].second;
  cout << t.str(true) << endl;
}
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
dolfin::linear_solver_methods()
{
  // Add default method
  std::vector<std::pair<std::string, std::string> >
    methods;
  methods.push_back(std::make_pair("default", "default linear solver"));

  // Add LU methods
  std::vector<std::pair<std::string, std::string> >
    lu_methods = DefaultFactory::factory().lu_solver_methods();
  for (unsigned int i = 0; i < lu_methods.size(); i++)
  {
    if (lu_methods[i].first != "default")
      methods.push_back(lu_methods[i]);
  }

  // Add Krylov methods
  std::vector<std::pair<std::string, std::string> >
    krylov_methods = DefaultFactory::factory().krylov_solver_methods();
  for (unsigned int i = 0; i < krylov_methods.size(); i++)
  {
    if (krylov_methods[i].first != "default")
      methods.push_back(krylov_methods[i]);
  }

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
bool dolfin::has_lu_solver_method(std::string method)
{
  std::vector<std::pair<std::string, std::string> > methods =
    DefaultFactory::factory().lu_solver_methods();
  for (unsigned int i = 0; i < methods.size(); i++)
    if (methods[i].first == method)
      return true;
  return false;
}
//-----------------------------------------------------------------------------
bool dolfin::has_krylov_solver_method(std::string method)
{
  std::vector<std::pair<std::string, std::string> > methods =
    DefaultFactory::factory().krylov_solver_methods();
  for (unsigned int i = 0; i < methods.size(); i++)
    if (methods[i].first == method)
      return true;
  return false;
}
//-----------------------------------------------------------------------------
bool dolfin::has_krylov_solver_preconditioner(std::string preconditioner)
{
  std::vector<std::pair<std::string, std::string> > preconditioners =
    DefaultFactory::factory().krylov_solver_preconditioners();
  for (unsigned int i = 0; i < preconditioners.size(); i++)
    if (preconditioners[i].first == preconditioner)
      return true;
  return false;
}
//-----------------------------------------------------------------------------
double dolfin::residual(const GenericLinearOperator& A,
                        const GenericVector& x,
                        const GenericVector& b)
{
  boost::shared_ptr<GenericVector> y = x.factory().create_vector();
  A.mult(x, *y);
  *y -= b;
  return y->norm("l2");
}
//-----------------------------------------------------------------------------
double dolfin::norm(const GenericVector& x, std::string norm_type)
{
  return x.norm(norm_type);
}
//-----------------------------------------------------------------------------
double dolfin::normalize(GenericVector& x, std::string normalization_type)
{
  if (x.empty())
  {
    dolfin_error("solve.cpp",
                 "normalize vector",
                 "Cannot normalize vector of zero length");
  }

  double c = 0.0;
  if (normalization_type == "l2")
  {
    c = x.norm("l2");
    x /= c;
  }
  else if (normalization_type == "average")
  {
    c = x.sum()/static_cast<double>(x.size());
    x -= c;
  }
  else
  {
    dolfin_error("solve.cpp",
                 "normalize vector",
                 "Unknown normalization type (\"%s\")",
                 normalization_type.c_str());
  }

  return c;
}
//-----------------------------------------------------------------------------
bool dolfin::has_linear_algebra_backend(std::string backend)
{
  if (backend == "uBLAS")
  {
    return true;
  }
  else if (backend == "PETSc")
  {
#ifdef HAS_PETSC
    return true;
#else
    return false;
#endif
  }
  else if (backend == "PETScCusp")
  {
#ifdef HAS_PETSC_CUSP
    return true;
#else
    return false;
#endif
  }
  else if (backend == "Epetra")
  {
#ifdef HAS_TRILINOS
    return true;
#else
    return false;
#endif
  }
  else if (backend == "STL")
  {
    return true;
  }
  return false;
}
//-------------------------------------------------------------------------
void dolfin::list_linear_algebra_backends()
{
  std::vector<std::pair<std::string, std::string> > backends =
    linear_algebra_backends();

  // Pretty-print list of available linear algebra backends
  Table t("Linear algebra backends", false);
  for (unsigned int i = 0; i < backends.size(); i++)
    t(backends[i].first, "Description") = backends[i].second;

  cout << t.str(true) << endl;
}
//-------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> > dolfin::linear_algebra_backends()
{
  std::vector<std::pair<std::string, std::string> > backends;

  std::map<std::string, std::string> default_backend
    = boost::assign::map_list_of("uBLAS",  " (default)")
                                ("Epetra", "")
                                ("PETSc", "");
  #ifdef HAS_PETSC
  default_backend["uBLAS"] = "";
  default_backend["PETSc"] = " (default)";
  #else
    #ifdef HAS_TRILINOS
    default_backend["uBLAS"] = "";
    default_backend["Epetra"] = " (default)";
    #endif
  #endif

  // Add available backends
  backends.push_back(std::make_pair("uBLAS", "Template based basic linear algebra "
				    "from boost" + default_backend["uBLAS"]));
  backends.push_back(std::make_pair("STL", "Light weight storage backend for Tensors"));

  #ifdef HAS_PETSC
  backends.push_back(std::make_pair("PETSc", "Powerful MPI parallel linear algebra"
				    " library" + default_backend["PETSc"]));
  #endif
  #ifdef HAS_PETSC_CUSP
  backends.push_back(std::make_pair("PETScCusp", "GPU-accelerated build of PETSc"));
  #endif
  #ifdef HAS_TRILINOS
  backends.push_back(std::make_pair("Epetra", "Powerful MPI parallel linear algebra"
				    " library" + default_backend["Epetra"]));
  #endif

  return backends;
}
//-------------------------------------------------------------------------
