// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-07-23
// Last changed: 

#ifndef __UBLAS_PRECONDITIONER_H
#define __UBLAS_PRECONDITIONER_H

#include <dolfin/ublas.h>
#include <dolfin/Array.h>
#include <dolfin/Parametrized.h>
#include <dolfin/uBlasSparseMatrix.h>

namespace dolfin
{

  /// This class provides a precondtioner for Krylov methods which operate
  /// on uBlas data types.

  /// FIXME: This class should be generalised. At the moment it is only for ILU
  ///        and users cannot provide their own preconditioner

  class DenseVector;

  class uBlasPreconditioner : public Parametrized
  {
  public:

    // Preconditioners
    enum Type
    { 
      default_pc, // Default preconditioner
      ilu,        // Incomplete LU (ILU(0))
      none        // No preconditioning
    };

    /// Constructor
    uBlasPreconditioner();

    /// Initialise preconditioner
    uBlasPreconditioner(const uBlasSparseMatrix& A);

    /// Destructor
    virtual ~uBlasPreconditioner();

    /// Initialise preconditioner
    void init(const uBlasSparseMatrix& A);

    /// Solve linear system (M^-1)Ax  (in-place)
    void solve(ublas_vector& x) const;

    /// Solve linear system (M^-1)Ax = y
    void solve(const ublas_vector& x, ublas_vector& y) const;

  private:

    // Preconditioner matrix
    uBlasSparseMatrix M;

    Array<uint> diagonal;

    // Create preconditioner matrix
    void create(const uBlasSparseMatrix& A);

  };

}

#endif
