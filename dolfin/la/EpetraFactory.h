// Copyright (C) 2008 Martin Sandve Alnes, Kent-Andre Mardal and Johannes Ring
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
// Modified by Anders Logg 2011
//
// First added:  2008-04-21
// Last changed: 2011-10-19

#ifdef HAS_TRILINOS

#ifndef __EPETRA_FACTORY_H
#define __EPETRA_FACTORY_H

#include <string>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include "EpetraKrylovSolver.h"
#include "EpetraLUSolver.h"
#include "EpetraMatrix.h"
#include "EpetraVector.h"
#include "SparsityPattern.h"
#include "LinearAlgebraFactory.h"

// Forwad declarations
class Epetra_MpiComm;
class Epetra_SerialComm;

namespace dolfin
{

  class GenericLinearSolver;

  class EpetraFactory : public LinearAlgebraFactory
  {
  public:

    /// Destructor
    virtual ~EpetraFactory();

    /// --- LinearAlgebraFactory interface

    /// Create empty matrix
    boost::shared_ptr<GenericMatrix> create_matrix() const;

    /// Create empty vector (global)
    boost::shared_ptr<GenericVector> create_vector() const;

    /// Create empty vector (local)
    boost::shared_ptr<GenericVector> create_local_vector() const;

    /// Create empty sparsity pattern
    boost::shared_ptr<GenericSparsityPattern> create_pattern() const;

    /// Create LU solver
    boost::shared_ptr<GenericLUSolver> create_lu_solver(std::string method) const;

    /// Create Krylov solver
    boost::shared_ptr<GenericLinearSolver>
      create_krylov_solver(std::string method,
                           std::string preconditioner) const;

    /// Return a list of available LU solver methods
    std::vector<std::pair<std::string, std::string> >
    lu_solver_methods() const;

    /// Return a list of available Krylov solver methods
    std::vector<std::pair<std::string, std::string> >
    krylov_solver_methods() const;

    /// Return a list of available preconditioners
    std::vector<std::pair<std::string, std::string> >
    krylov_solver_preconditioners() const;

    /// --- EpetraFactory interface

    // Return Epetra Communicator
    Epetra_SerialComm& get_serial_comm();

    // Return Epetra Communicator
    Epetra_MpiComm& get_mpi_comm();

    // Return singleton instance
    static EpetraFactory& instance()
    { return factory; }

  private:

    // Private constructor
    EpetraFactory();

    // Singleton instance
    static EpetraFactory factory;

    // Communicator
    boost::scoped_ptr<Epetra_SerialComm> serial_comm;

    // Communicator
    boost::scoped_ptr<Epetra_MpiComm> mpi_comm;

  };

}

#endif

#endif
