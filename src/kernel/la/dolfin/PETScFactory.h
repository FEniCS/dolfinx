// Copyright (C) 2007 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-11-30
// Last changed: 2007-11-30


#ifndef __ PETSC_FACTORY_H
#define __ PETSC_FACTORY_H

#include <dolfin/PETScMatrix.h>
#include <dolfin/PETScVector.h>
#include <dolfin/SparsityPattern.h>

namespace dolfin
{

  class PETScFactory: public LinearAlgebraFactory
  {
    public:

    /// Constructor
    PETScFactory() {}

    /// Destructor
    virtual ~PETScFactory() {}

    /// Create empty matrix
    virtual PETScMatrix* createMatrix() const { return new PETScMatrix(); }

    /// Create empty sparsity pattern 
    virtual SparsityPattern * createPattern() const {return new SparsityPattern(); }

    /// Create empty vector
    virtual PETScVector* createVector() const { return new PETScVector(); }

  };

}

#
