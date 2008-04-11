// Copyright (C) 2008 Ola Skavhaug
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-04-11

#include <dolfin/log/log.h>
#include "GenericMatrix.h"
#include "GenericVector.h"
#include "uBlasLinearSolver.h"
#include "uBlasMatrix.h"

namespace dolfin {

  uint uBlasLinearSolver::solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b) 
  {
    uBlasVector* xx = dynamic_cast<uBlasVector*>(x.instance());
    if (!xx) error("Could not convert second arument to a uBlasVector");
    const uBlasVector* bb = dynamic_cast<const uBlasVector*>(b.instance());
    if (!bb) error("Could not convert third arument to a uBlasVector");

    const uBlasMatrix<ublas_dense_matrix>* AD = dynamic_cast<const uBlasMatrix<ublas_dense_matrix>*>(A.instance());
    if (AD)
      return solve(*AD, *xx, *bb);

    const uBlasMatrix<ublas_sparse_matrix>* AS = dynamic_cast<const uBlasMatrix<ublas_sparse_matrix>*>(A.instance());
    if (AS)
      return solve(*AD, *xx, *bb);

    error("Could not convert first argument to a uBlasMatrix<>");
    return 0;
  }

}
