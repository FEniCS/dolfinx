// Copyright (C) 2007 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-11-30
// Last changed: 2007-11-30

#ifdef HAVE_PETSC_H

#ifndef __PETSC_FACTORY_H
#define __PETSC_FACTORY_H


#include <dolfin/LinearAlgebraFactory.h>

namespace dolfin
{

  class PETScFactory: public LinearAlgebraFactory
  {
  public:

    /// Destructor
    virtual ~PETScFactory() {}

    /// Create empty matrix
    GenericMatrix* createMatrix() const;

    /// Create empty sparsity pattern 
    GenericSparsityPattern* createPattern() const;

    /// Create empty vector
    GenericVector* createVector() const;

    static PETScFactory& instance() { return petscfactory; }

  private:
    /// Private Constructor
    PETScFactory() {}
    static PETScFactory petscfactory;

  };

}

#endif

#endif
