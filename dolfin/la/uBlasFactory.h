// Copyright (C) 2007 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-12-06
// Last changed: 2007-12-06


#ifndef __UBLAS_FACTORY_H
#define __UBLAS_FACTORY_H

#include "uBlasMatrix.h"
#include "uBlasVector.h"
#include "SparsityPattern.h"
#include "LinearAlgebraFactory.h"

namespace dolfin
{

  class uBlasFactory: public LinearAlgebraFactory
  {
  public:

    /// Destructor
    virtual ~uBlasFactory() {}

    /// Create empty matrix
    uBlasMatrix<ublas_sparse_matrix>* createMatrix() const;

    /// Create empty sparsity pattern 
    SparsityPattern* createPattern() const;

    /// Create empty vector
    uBlasVector* createVector() const;

    /// Return sigleton instance
    static uBlasFactory& instance() { return ublasfactory; }

  private:
    /// Private Constructor
    uBlasFactory() {}

    static uBlasFactory ublasfactory;

  };

}

#endif
