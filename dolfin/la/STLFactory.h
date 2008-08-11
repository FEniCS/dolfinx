// Copyright (C) 2007 Ilmar Wilbers.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
//
// First added:  2008-05-21
// Last changed: 2008-08-07

#ifndef __STL_FACTORY_H
#define __STL_FACTORY_H

#include "STLMatrix.h"
#include "uBlasVector.h"
#include "GenericSparsityPattern.h"
#include "LinearAlgebraFactory.h"

namespace dolfin
{

  class STLFactory: public LinearAlgebraFactory
  {
  public:

    /// Destructor
    virtual ~STLFactory() {}

    /// Create empty matrix
    STLMatrix* createMatrix() const
    { return new STLMatrix(); }

    /// Create empty vector
    uBlasVector* createVector() const
    { return new uBlasVector(); }

    /// Create empty sparsity pattern 
    GenericSparsityPattern* createPattern() const
    { return 0; }

    /// Return singleton instance
    static STLFactory& instance()
    { return factory; }

  private:

    /// Private Constructor
    STLFactory() {}

    // Singleton instance
    static STLFactory factory;

  };
}

#endif
