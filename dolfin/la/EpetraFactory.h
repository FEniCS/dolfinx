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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2008-04-21
// Last changed: 2008-09-28

#ifdef HAS_TRILINOS

#ifndef __EPETRA_FACTORY_H
#define __EPETRA_FACTORY_H

#include <string>
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
    EpetraMatrix* create_matrix() const;

    /// Create empty vector (global)
    EpetraVector* create_vector() const;

    /// Create empty vector (local)
    EpetraVector* create_local_vector() const;

    /// Create empty sparsity pattern
    SparsityPattern* create_pattern() const;

    /// Create LU solver
    EpetraLUSolver* create_lu_solver() const;

    /// Create Krylov solver
    EpetraKrylovSolver* create_krylov_solver(std::string method,
                                             std::string pc) const;

    /// --- EpetraFactory interface

    // Return Epetra Communicator
    Epetra_SerialComm& get_serial_comm() const;

    // Return Epetra Communicator
    Epetra_MpiComm& get_mpi_comm() const;

    // Return singleton instance
    static EpetraFactory& instance()
    { return factory; }

  private:

    // Private constructor
    EpetraFactory();

    // Singleton instance
    static EpetraFactory factory;

    // Communicator
    Epetra_SerialComm* serial_comm;

    // Communicator
    Epetra_MpiComm* mpi_comm;

  };

}

#endif

#endif
