// Copyright (C) 2008 Martin Sandve Alnes, Kent-Andre Mardal and Johannes Ring.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-04-21

#ifdef HAS_TRILINOS

#ifndef __EPETRA_FACTORY_H
#define __EPETRA_FACTORY_H

#include "EpetraMatrix.h"
#include "EpetraVector.h"
#include "EpetraSparsityPattern.h"
#include "LinearAlgebraFactory.h"
//#include "Epetra_SerialComm.h"

class Epetra_SerialComm; 

namespace dolfin
{

  class EpetraFactory : public LinearAlgebraFactory
  {
  public:

    /// Destructor
    virtual ~EpetraFactory();

    /// Create empty matrix
    EpetraMatrix* create_matrix() const;

    /// Create empty vector
    EpetraVector* create_vector() const;

    /// Create empty sparsity pattern 
    EpetraSparsityPattern* create_pattern() const;

    // Return Epetra Communicator  
    Epetra_SerialComm& getSerialComm(); 

    // Return singleton instance
    static EpetraFactory& instance()
    { return factory; }

  private:
    
    // Private constructor
    EpetraFactory();

    // Singleton instance
    static EpetraFactory factory;

    // Communicator
    Epetra_SerialComm* comm;

  };

}

#endif

#endif
