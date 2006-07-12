// Copyright (C) 2004-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2004-06-19
// Last changed: 2006-06-07

#ifndef __LINEAR_SOLVER_H
#define __LINEAR_SOLVER_H

#include <dolfin/dolfin_log.h>
#include <dolfin/ublas.h>

namespace dolfin
{

  /// Forward declarations
  class DenseVector;  
  template<class Mat>
  class uBlasMatrix;


#ifdef HAVE_PETSC_H
  class PETScSparseMatrix;
  class PETScVector;
  class VirtualMatrix;
#endif


  /// This class defines the interfaces for linear solvers for
  /// systems of the form Ax = b.
  class LinearSolver
  {
  public:

    /// Constructor
    LinearSolver();

    /// Destructor
    virtual ~LinearSolver();

    /// Solve linear system Ax = b (uBlas sparse matrix version)
    virtual uint solve(const uBlasMatrix<ublas_sparse_matrix>& A, DenseVector& x, const DenseVector& b);

#ifdef HAVE_PETSC_H
    /// Solve linear system Ax = b (PETSc sparse matrix version)
    virtual uint solve(const PETScSparseMatrix& A, PETScVector& x, const PETScVector& b);

    /// Solve linear system Ax = b (PETSc matrix-free version)
    virtual uint solve(const VirtualMatrix& A, PETScVector& x, const PETScVector& b);
#endif

  };

}

#endif
