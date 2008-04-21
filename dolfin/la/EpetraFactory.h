// Copyright (C) 2008 Kent-Andre Mardal and Johannes Ring.
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
#include "Epetra_SerialComm.h"

namespace dolfin
{

  class EpetraFactory : public LinearAlgebraFactory
  {
  public:

    /// Destructor
    virtual ~EpetraFactory() {}

    /// Create empty matrix
    EpetraMatrix* createMatrix() const;

    /// Create empty sparsity pattern 
    EpetraSparsityPattern* createPattern() const;

    /// Create empty vector
    EpetraVector* createVector() const;

    // Return Epetra Communicator  
    Epetra_SerialComm& getSerialComm() { return comm;}; 

    static EpetraFactory& instance() { return epetrafactory; }

  private:

    /// Private Constructor
    EpetraFactory() {}
    static EpetraFactory epetrafactory;
    Epetra_SerialComm comm;

  };

}

#endif

#endif
