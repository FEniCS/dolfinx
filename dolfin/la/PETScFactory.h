// Copyright (C) 2007 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-11-30
// Last changed: 2007-12-06

#ifdef HAS_PETSC

#ifndef __PETSC_FACTORY_H
#define __PETSC_FACTORY_H

#include "PETScLUSolver.h"
#include "PETScMatrix.h"
#include "PETScVector.h"
#include "SparsityPattern.h"
#include "LinearAlgebraFactory.h"

namespace dolfin
{

  class PETScFactory : public LinearAlgebraFactory
  {
  public:

    /// Destructor
    virtual ~PETScFactory() {}

    /// Create empty matrix
    PETScMatrix* create_matrix() const;

    /// Create empty vector (global)
    PETScVector* create_vector() const;

    /// Create empty vector (local)
    PETScVector* create_local_vector() const;

    /// Create empty sparsity pattern
    SparsityPattern* create_pattern() const;

    /// Create LU solver
    PETScLUSolver* create_lu_solver() const;

    /// Return singleton instance
    static PETScFactory& instance()
    { return factory; }

  private:

    /// Private constructor
    PETScFactory() {}
    static PETScFactory factory;

  };

}

#endif

#endif
