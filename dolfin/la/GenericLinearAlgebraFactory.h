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
// Last changed: 2012-08-22

#ifndef __GENERIC_LINEAR_ALGEBRA_FACTORY_H
#define __GENERIC_LINEAR_ALGEBRA_FACTORY_H

#include <string>
#include <vector>
#include <memory>
#include <dolfin/common/MPI.h>
#include <dolfin/common/types.h>
#include <dolfin/log/log.h>

// Included here so we can define dummy class below
#include "GenericLinearOperator.h"

namespace dolfin
{

  class GenericLinearSolver;
  class GenericLUSolver;
  class GenericMatrix;
  class GenericVector;
  class TensorLayout;

  class GenericLinearAlgebraFactory
  {
  public:

    /// Constructor
    GenericLinearAlgebraFactory() {}

    /// Destructor
    virtual ~GenericLinearAlgebraFactory() {}

    /// Create empty matrix
    virtual std::shared_ptr<GenericMatrix> create_matrix(MPI_Comm comm) const = 0;

    /// Create empty vector
    virtual std::shared_ptr<GenericVector>
      create_vector(MPI_Comm comm) const = 0;

    /// Create empty tensor layout
    virtual std::shared_ptr<TensorLayout> create_layout(std::size_t rank) const = 0;

    /// Create empty linear operator
    virtual std::shared_ptr<GenericLinearOperator>
    create_linear_operator() const = 0;

    /// Create LU solver
    virtual std::shared_ptr<GenericLUSolver>
    create_lu_solver(MPI_Comm comm, std::string method) const = 0;

    /// Create Krylov solver
    virtual std::shared_ptr<GenericLinearSolver>
    create_krylov_solver(MPI_Comm comm,
                         std::string method,
                         std::string preconditioner) const = 0;

    /// Return a list of available LU solver methods.  This function
    /// should be overloaded by subclass if non-empty.
    virtual std::map<std::string, std::string> lu_solver_methods() const
    { return std::map<std::string, std::string>(); }

    /// Return a list of available Krylov solver methods.  This
    /// function should be overloaded by subclass if non-empty.
    virtual std::map<std::string, std::string> krylov_solver_methods() const
    { return std::map<std::string, std::string>(); }

    /// Return a list of available preconditioners.
    /// This function should be overloaded by subclass if non-empty.
    virtual std::map<std::string, std::string>
    krylov_solver_preconditioners() const
    { return std::map<std::string, std::string>(); }

  protected:

    // Dummy class that can be returned for linear algebra backends
    // that do not support the GenericLinearOperator interface
    class NotImplementedLinearOperator : public GenericLinearOperator
    {
    public:

      std::size_t size(std::size_t dim) const
      { return 0; }

      void mult(const GenericVector& x, GenericVector& y) const
      { dolfin_not_implemented(); }

      std::string str(bool verbose) const
      { dolfin_not_implemented(); return ""; }

    };

  };

}

#endif
