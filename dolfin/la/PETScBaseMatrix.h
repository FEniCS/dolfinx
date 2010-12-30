// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-01-17
// Last changed: 2009-09-08

#ifndef __PETSC_BASE_MATRIX_H
#define __PETSC_BASE_MATRIX_H

#ifdef HAS_PETSC

#include <string>
#include <boost/shared_ptr.hpp>
#include <petscmat.h>
#include <dolfin/common/types.h>
#include <dolfin/common/Variable.h>
#include <dolfin/log/dolfin_log.h>
#include "PETScObject.h"

namespace dolfin
{

  class PETScMatrixDeleter
  {
  public:
    void operator() (Mat* A)
    {
      if (A)
        MatDestroy(*A);
      delete A;
    }
  };

  /// This class is a base class for matrices that can be used in
  /// PETScKrylovSolver.

  class PETScBaseMatrix : public PETScObject, public virtual Variable
  {
  public:

    /// Constructor
    PETScBaseMatrix() {}

    /// Constructor
    PETScBaseMatrix(boost::shared_ptr<Mat> A) : A(A) {}

    /// Resize virtual matrin
    virtual void resize(uint m, uint n) = 0;

    /// Return number of rows (dim = 0) or columns (dim = 1) along dimension dim
    uint size(uint dim) const
    {
      assert(dim <= 1);
      if (A)
      {
        int m(0), n(0);
        MatGetSize(*A, &m, &n);
        if (dim == 0)
          return m;
        else
          return n;
      }
      else
        return 0;
    }

    /// Return local rang along dimension dim
    std::pair<uint, uint> local_range(uint dim) const
    {
      assert(dim <= 1);
      if (dim == 1)
        error("Cannot compute columns range for PETSc matrices.");
      if (A)
      {
        int m(0), n(0);
        MatGetOwnershipRange(*A, &m, &n);
        return std::make_pair(m, n);
      }
      else
        return std::make_pair(0, 0);
    }

    /// Return PETSc Mat pointer
    boost::shared_ptr<Mat> mat() const
    { return A; }

    /// Return informal string representation (pretty-print)
    virtual std::string str(bool verbose) const = 0;

  protected:

    // PETSc Mat pointer
    boost::shared_ptr<Mat> A;
  };

}

#endif

#endif
