// Copyright (C) 2007 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-11-30
// Last changed: 2007-11-30


#ifndef __ LINEAR_ALGEBRA_FACTORY_H
#define __ LINEAR_ALGEBRA_FACTORY_H

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
