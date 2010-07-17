// Copyright (C) 2008 Martin Sandve Alnes, Kent-Andre Mardal and Johannes Ring.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-04-21
// Last changed: 2008-09-28

#ifdef HAS_TRILINOS

#ifndef __EPETRA_FACTORY_H
#define __EPETRA_FACTORY_H

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
