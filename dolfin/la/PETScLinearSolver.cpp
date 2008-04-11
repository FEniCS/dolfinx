// Copyright (C) 2008 Ola Skavhaug
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-04-11

#ifdef HAS_PETSC

#include "GenericMatrix.h"
#include "GenericVector.h"
#include "PETScMatrix.h"
#include "PETScVector.h"
#include "PETScLinearSolver.h"

namespace dolfin
{
  int PETScLinearSolver:: solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b)
  {
    const PETScMatrix* AA = dynamic_cast<const PETScMatrix*>(A.instance());
    if (!AA) error("Could not convert first arguement to a PETScMatrix");
    PETScVector* xx = dynamic_cast<PETScVector*>(x.instance());
    if (!xx) error("Could not convert second arguement to a PETScVector");
    const PETScVector* bb = dynamic_cast<const PETScVector*>(b.instance());
    if (!bb) error("Could not convert third arguement to a PETScVector");

    return solve(*AA, *xx, *bb);
  }
}

#endif
