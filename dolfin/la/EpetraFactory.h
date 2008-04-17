// Copyright (C) 2008 Johannes Ring.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-01-24
// Last changed: 2008-01-24

#ifdef HAS_TRILINOS

#ifndef __EPETRA_FACTORY_H
#define __EPETRA_FACTORY_H

#include "EpetraMatrix.h"
#include "EpetraVector.h"
#include "SparsityPattern.h"
#include "LinearAlgebraFactory.h"

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
    SparsityPattern* createPattern() const;

    /// Create empty vector
    EpetraVector* createVector() const;

    static EpetraFactory& instance() { return epetrafactory; }

  private:

    /// Private Constructor
    EpetraFactory() {}
    static EpetraFactory epetrafactory;

  };

}

#endif

#endif
