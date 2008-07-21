// Copyright (C) 2008 Dag Lindbo
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-07-06

#ifdef HAS_MTL4

#ifndef __MTL4_FACTORY_H
#define __MTL4_FACTORY_H

#include "MTL4Matrix.h"
#include "MTL4Vector.h"
#include "LinearAlgebraFactory.h"
#include "MTL4SparsityPattern.h"

namespace dolfin
{
  class MTL4Factory : public LinearAlgebraFactory
  {
  public:

    /// Destructor
    virtual ~MTL4Factory() {}

    /// Create empty matrix
    MTL4Matrix* createMatrix() const;

    /// Create empty vector
    MTL4Vector* createVector() const;

    /// Dummy sparsity pattern
    MTL4SparsityPattern* createPattern() const;

    // Return singleton instance
    static MTL4Factory& instance()
    { return factory; }

  private:
    
    // Private constructor
    MTL4Factory();

    // Singleton instance
    static MTL4Factory factory;
  };
}

#endif
#endif
