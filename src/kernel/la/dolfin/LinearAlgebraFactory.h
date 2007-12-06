// Copyright (C) 2007 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-11-30
// Last changed: 2007-11-30

#ifndef __LINEAR_ALGEBRA_FACTORY_H
#define __LINEAR_ALGEBRA_FACTORY_H

#include <dolfin/GenericMatrix.h>
#include <dolfin/GenericSparsityPattern.h>
#include <dolfin/GenericVector.h>

namespace dolfin
{

  class LinearAlgebraFactory
  {
    public:

    /// Constructor
    LinearAlgebraFactory() {}

    /// Destructor
    virtual ~LinearAlgebraFactory() {}

    /// Create empty matrix
    virtual GenericMatrix* createMatrix() const = 0;

    /// Create empty sparsity pattern 
    virtual GenericSparsityPattern * createPattern() const = 0;

    /// Create empty vector
    virtual GenericVector* createVector() const = 0;

  };

}

#endif
