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
#include "EigenFactory.h"
#include "PETScFactory.h"
#include "STLFactory.h"
#include "TpetraFactory.h"
#include "DefaultFactory.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::shared_ptr<GenericMatrix> DefaultFactory::create_matrix() const
{
  return factory().create_matrix();
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericVector> DefaultFactory::create_vector(MPI_Comm comm) const
{
  return factory().create_vector(comm);
}
//-----------------------------------------------------------------------------
std::shared_ptr<TensorLayout>
DefaultFactory::create_layout(std::size_t rank) const
{
  return factory().create_layout(rank);
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericLinearOperator>
DefaultFactory::create_linear_operator() const
{
  return factory().create_linear_operator();
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericLUSolver>
  DefaultFactory::create_lu_solver(std::string method) const
{
  return factory().create_lu_solver(method);
}
//-----------------------------------------------------------------------------
std::shared_ptr<GenericLinearSolver>
DefaultFactory::create_krylov_solver(std::string method,
                                     std::string preconditioner) const
{
  return factory().create_krylov_solver(method, preconditioner);
}
//-----------------------------------------------------------------------------
std::map<std::string, std::string> DefaultFactory::lu_solver_methods() const
{
  return factory().lu_solver_methods();
}
 //-----------------------------------------------------------------------------
std::map<std::string, std::string>
DefaultFactory::krylov_solver_methods() const
{
  return factory().krylov_solver_methods();
}
//-----------------------------------------------------------------------------
std::map<std::string, std::string>
DefaultFactory::krylov_solver_preconditioners() const
{
  return factory().krylov_solver_preconditioners();
}
//-----------------------------------------------------------------------------
GenericLinearAlgebraFactory& DefaultFactory::factory()
{
  // Fallback
  const std::string default_backend = "Eigen";

  // Get backend from parameter system
  const std::string backend = dolfin::parameters["linear_algebra_backend"];

  // Choose backend
  if (backend == "Eigen")
    return EigenFactory::instance();
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
  else if (backend == "STL")
    return STLFactory::instance();
  else if (backend == "Tpetra")
  {
    #ifdef HAS_TRILINOS
    return TpetraFactory::instance();
    #else
    dolfin_error("DefaultFactory.cpp",
                 "access linear algebra backend",
                 "Tpetra linear algebra backend is not available");
    #endif
  }

  // Fallback
  log(WARNING, "Linear algebra backend \"" + backend
      + "\" not available, using " + default_backend + ".");
  return EigenFactory::instance();
}
//-----------------------------------------------------------------------------
