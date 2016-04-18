// Copyright (C) 2007 Ola Skavhaug
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
// Modified by Anders Logg 2011-2012
//
// First added:  2007-11-30
// Last changed: 2012-08-20

#ifdef HAS_PETSC

#ifndef __PETSC_FACTORY_H
#define __PETSC_FACTORY_H

#include <string>
#include <memory>
#include <dolfin/common/types.h>
#include "PETScKrylovSolver.h"
#include "PETScLUSolver.h"
#include "PETScMatrix.h"
#include "PETScVector.h"
#include "TensorLayout.h"
#include "GenericLinearAlgebraFactory.h"

namespace dolfin
{

  class PETScFactory : public GenericLinearAlgebraFactory
  {
  public:

    /// Destructor
    virtual ~PETScFactory() {}

    /// Create empty matrix
    std::shared_ptr<GenericMatrix> create_matrix(MPI_Comm comm) const;

    /// Create empty vector
    std::shared_ptr<GenericVector> create_vector(MPI_Comm comm) const;

    /// Create empty tensor layout
    std::shared_ptr<TensorLayout> create_layout(std::size_t rank) const;

    /// Create empty linear operator
    std::shared_ptr<GenericLinearOperator>
      create_linear_operator(MPI_Comm comm) const;

    /// Create LU solver
    std::shared_ptr<GenericLUSolver> create_lu_solver(MPI_Comm comm,
                                                      std::string method) const;

    /// Create Krylov solver
    std::shared_ptr<GenericLinearSolver>
    create_krylov_solver(MPI_Comm comm,
                         std::string method,
                         std::string preconditioner) const;

    /// Return a list of available LU solver methods
    std::map<std::string, std::string> lu_solver_methods() const;

    /// Return a list of available Krylov solver methods
    std::map<std::string, std::string> krylov_solver_methods() const;

    /// Return a list of available preconditioners
    std::map<std::string, std::string> krylov_solver_preconditioners() const;

    /// Return singleton instance
    static PETScFactory& instance()
    { return factory; }

  private:

    /// Private constructor
    PETScFactory() {}
    static PETScFactory factory;

  };

}

#endif

#endif
