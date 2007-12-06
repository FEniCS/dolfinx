// Copyright (C) 2007 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-12-6
// Last changed: 2007-12-6


#ifndef __UBLAS_FACTORY_H
#define __UBLAS_FACTORY_H

#include <dolfin/SparsityPattern.h>
#include <dolfin/LinearAlgebraFactory.h>
#include <dolfin/ublas.h>

namespace dolfin
{

  class uBlasFactory: public LinearAlgebraFactory
  {
  public:

    /// Destructor
    virtual ~uBlasFactory() {}

    /// Create empty matrix
    //uBlasMatrix<ublas_dense_matrix>* createMatrix() const;
    GenericMatrix* createMatrix() const;

    /// Create empty sparsity pattern 
    SparsityPattern* createPattern() const;

    /// Create empty vector
    GenericVector* createVector() const;

    static uBlasFactory& instance() { return ublasfactory; }

  private:
    /// Private Constructor
    uBlasFactory() {}

    static uBlasFactory ublasfactory;

  };

}

#endif
