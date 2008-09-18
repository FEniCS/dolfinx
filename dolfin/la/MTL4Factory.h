// Copyright (C) 2008 Dag Lindbo
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
//
// First added:  2008-07-06
// Last changed: 2008-08-11

#ifdef HAS_MTL4

#ifndef __MTL4_FACTORY_H
#define __MTL4_FACTORY_H

#include "MTL4Matrix.h"
#include "MTL4Vector.h"
#include "GenericSparsityPattern.h"
#include "LinearAlgebraFactory.h"

namespace dolfin
{
  class MTL4Factory : public LinearAlgebraFactory
  {
  public:



    /// Destructor
    virtual ~MTL4Factory() {}

    /// Create empty matrix
    MTL4Matrix* create_matrix() const
    { return new MTL4Matrix(); }

    /// Create empty vector
    MTL4Vector* create_vector() const
    { return new MTL4Vector(); }

    /// Dummy sparsity pattern
    GenericSparsityPattern* create_pattern() const
    { return 0; }

    // Return singleton instance
    static MTL4Factory& instance()
    { return factory; }

  private:
    
    // Private constructor
    MTL4Factory() {}

    // Singleton instance
    static MTL4Factory factory;

  };
}

#endif
#endif
