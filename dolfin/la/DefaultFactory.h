// Copyright (C) 2008-2012 Anders Logg
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
// First added:  2008-05-17
// Last changed: 2012-08-21

#ifndef __DEFAULT_FACTORY_H
#define __DEFAULT_FACTORY_H

#include <string>
#include <memory>
#include <dolfin/common/MPI.h>
#include <dolfin/common/types.h>
#include "GenericLinearAlgebraFactory.h"

namespace dolfin
{

  class DefaultFactory : public GenericLinearAlgebraFactory
  {

    class GenericLinearSolver;

    public:

    /// Constructor
    DefaultFactory() {}

    /// Destructor
    virtual ~DefaultFactory() {}

    /// Create empty matrix
    virtual std::shared_ptr<GenericMatrix> create_matrix() const;

    /// Create empty vector
    virtual std::shared_ptr<GenericVector> create_vector(MPI_Comm comm) const;

    /// Create empty tensor layout
    virtual std::shared_ptr<TensorLayout> create_layout(std::size_t rank) const;

    /// Create empty linear operator
    virtual std::shared_ptr<GenericLinearOperator>
    create_linear_operator() const;

    /// Create LU solver
    virtual std::shared_ptr<dolfin::GenericLUSolver>
    create_lu_solver(std::string method) const;

    /// Create Krylov solver
    virtual std::shared_ptr<dolfin::GenericLinearSolver>
    create_krylov_solver(std::string method, std::string preconditioner) const;

    /// Return a list of available LU solver methods
    std::map<std::string, std::string> lu_solver_methods() const;

    /// Return a list of available Krylov solver methods
    std::map<std::string, std::string> krylov_solver_methods() const;

    /// Return a list of available preconditioners
    std::map<std::string, std::string> krylov_solver_preconditioners() const;

    /// Return instance of default backend
    static GenericLinearAlgebraFactory& factory();

  };

}

#endif
