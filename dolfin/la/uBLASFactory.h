// Copyright (C) 2007 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2008.
//
// First added:  2007-12-06
// Last changed: 2008-05-18

#ifndef __UBLAS_FACTORY_H
#define __UBLAS_FACTORY_H

#include "uBLASMatrix.h"
#include "uBLASVector.h"
#include "SparsityPattern.h"
#include "LinearAlgebraFactory.h"

namespace dolfin
{
  // Forward declaration
  template<class T> class uBLASMatrix;

  template<class Mat = ublas_sparse_matrix>
  class uBLASFactory: public LinearAlgebraFactory
  {
  public:

    /// Destructor
    virtual ~uBLASFactory() {}

    /// Create empty matrix
    uBLASMatrix<Mat>* createMatrix() const
    { return new uBLASMatrix<Mat>(); }

    /// Create empty sparsity pattern 
    SparsityPattern* createPattern() const
    { return new SparsityPattern(); }

    /// Create empty vector
    uBLASVector* createVector() const
    { return new uBLASVector(); }

    static uBLASFactory<Mat>& instance() 
    { return factory; }

  private:

    // Private Constructor
    uBLASFactory() {}

    // Singleton instance
    static uBLASFactory<Mat> factory;
  };
}

// Initialise static data
template<class Mat> dolfin::uBLASFactory<Mat> dolfin::uBLASFactory<Mat>::factory;

#endif
