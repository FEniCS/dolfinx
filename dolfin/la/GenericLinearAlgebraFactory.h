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
#include <boost/shared_ptr.hpp>
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
    virtual boost::shared_ptr<GenericMatrix> create_matrix() const = 0;

    /// Create empty vector (global)
    virtual boost::shared_ptr<GenericVector> create_vector() const = 0;

    /// Create empty vector (local)
    virtual boost::shared_ptr<GenericVector> create_local_vector() const = 0;

    /// Create empty tensor layout
    virtual boost::shared_ptr<TensorLayout> create_layout(uint rank) const = 0;

    /// Create empty Krylov matrix
    virtual boost::shared_ptr<GenericLinearOperator> create_linear_operator() const = 0;

    /// Create LU solver
    virtual boost::shared_ptr<GenericLUSolver>
    create_lu_solver(std::string method) const = 0;

    /// Create Krylov solver
    virtual boost::shared_ptr<GenericLinearSolver>
    create_krylov_solver(std::string method, std::string preconditioner) const = 0;

    /// Return a list of available LU solver methods.
    /// This function should be overloaded by subclass if non-empty.
    virtual std::vector<std::pair<std::string, std::string> >
    lu_solver_methods() const
    {
      std::vector<std::pair<std::string, std::string> > methods;
      return methods;
    }

    /// Return a list of available Krylov solver methods.
    /// This function should be overloaded by subclass if non-empty.
    virtual std::vector<std::pair<std::string, std::string> >
    krylov_solver_methods() const
    {
      std::vector<std::pair<std::string, std::string> > methods;
      return methods;
    }

    /// Return a list of available preconditioners.
    /// This function should be overloaded by subclass if non-empty.
    virtual std::vector<std::pair<std::string, std::string> >
    krylov_solver_preconditioners() const
    {
      std::vector<std::pair<std::string, std::string> > preconditioners;
      return preconditioners;
    }

  protected:

    // Dummy class that can be returned for linear algebra backends
    // that do not support the GenericLinearOperator interface
    class NotImplementedLinearOperator : public GenericLinearOperator
    {
    public:

      uint size(uint dim) const
      { return 0; }

      void mult(const GenericVector& x, GenericVector& y) const
      { dolfin_not_implemented(); }

      std::string str(bool verbose) const
      { dolfin_not_implemented(); return ""; }

    };

  };

}

#endif
