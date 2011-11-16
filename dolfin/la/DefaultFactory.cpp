// Copyright (C) 2008-2011 Anders Logg
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
// Modified by Fredrik Valdmanis, 2011
//
// First added:  2008-05-17
// Last changed: 2011-11-11

#include <dolfin/parameter/GlobalParameters.h>
#include "uBLASFactory.h"
#include "PETScFactory.h"
#include "PETScCuspFactory.h"
#include "EpetraFactory.h"
#include "MTL4Factory.h"
#include "STLFactory.h"
#include "DefaultFactory.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
GenericMatrix* DefaultFactory::create_matrix() const
{
  return factory().create_matrix();
}
//-----------------------------------------------------------------------------
GenericVector* DefaultFactory::create_vector() const
{
  return factory().create_vector();
}
//-----------------------------------------------------------------------------
GenericVector* DefaultFactory::create_local_vector() const
{
  return factory().create_local_vector();
}
//-----------------------------------------------------------------------------
GenericSparsityPattern* DefaultFactory::create_pattern() const
{
  return factory().create_pattern();
}
//-----------------------------------------------------------------------------
GenericLinearSolver* DefaultFactory::create_lu_solver(std::string method) const
{
  return factory().create_lu_solver(method);
}
//-----------------------------------------------------------------------------
GenericLinearSolver*
DefaultFactory::create_krylov_solver(std::string method,
                                     std::string preconditioner) const
{
  return factory().create_krylov_solver(method, preconditioner);
}
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
DefaultFactory::lu_solver_methods() const
{
  return factory().lu_solver_methods();
}
 //-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
DefaultFactory::krylov_solver_methods() const
{
  return factory().krylov_solver_methods();
}
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
DefaultFactory::krylov_solver_preconditioners() const
{
  return factory().krylov_solver_preconditioners();
}
//-----------------------------------------------------------------------------
LinearAlgebraFactory& DefaultFactory::factory()
{
  // Fallback
  const std::string default_backend = "uBLAS";
  typedef uBLASFactory<> DefaultFactory;

  // Get backend from parameter system
  const std::string backend = dolfin::parameters["linear_algebra_backend"];

  // Choose backend
  if (backend == "uBLAS")
  {
    return uBLASFactory<>::instance();
  }
  else if (backend == "PETSc")
  {
#ifdef HAS_PETSC
    return PETScFactory::instance();
#else
    dolfin_error("DefaultFactory.cpp",
                 "access linear algebra backend",
                 "PETSc linear algebra backend is not available");
#endif
  }
  else if (backend == "PETScCusp")
  {
#ifdef HAS_PETSC_CUSP
    return PETScCuspFactory::instance();
#else
    dolfin_error("DefaultFactory.cpp",
                 "access linear algebra backend",
                 "PETScCusp linear algebra backend is not available");
#endif
  }
  else if (backend == "Epetra")
  {
#ifdef HAS_TRILINOS
    return EpetraFactory::instance();
#else
    dolfin_error("DefaultFactory.cpp",
                 "access linear algebra backend",
                 "Trilinos linear algebra backend is not available");
#endif
  }
  else if (backend == "MTL4")
  {
#ifdef HAS_MTL4
    return MTL4Factory::instance();
#else
    dolfin_error("DefaultFactory.cpp",
                 "access linear algebra backend",
                 "MTL4 linear algebra backend is not available");
#endif
  }
  else if (backend == "STL")
  {
    return STLFactory::instance();
  }

  // Fallback
  log(WARNING, "Linear algebra backend \"" + backend + "\" not available, using " + default_backend + ".");
  return DefaultFactory::instance();
}
//-----------------------------------------------------------------------------
