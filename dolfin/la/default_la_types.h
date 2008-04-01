// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-07-03
// Last changed:

#ifndef __DEFAULT_LA_TYPES_H
#define __DEFAULT_LA_TYPES_H

#include "PETScVector.h"
#include "PETScMatrix.h"
#include "PETScLUSolver.h"
#include "PETScKrylovSolver.h"

#include "uBlasVector.h"
#include "uBlasSparseMatrix.h"
#include "uBlasLUSolver.h"
#include "uBlasKrylovSolver.h"

namespace dolfin
{

  /// Various default linear algebra quantities are defined here.
  
#ifdef HAS_PETSC
  typedef PETScVector         DefaultVector;
  typedef PETScMatrix         DefaultMatrix;
  typedef PETScLUSolver       DefaultLUSolver;
  typedef PETScKrylovSolver   DefaultKrylovSolver;
  typedef PETScPreconditioner DefaultPreconditioner; 
#else
  typedef uBlasVector         DefaultVector;
  typedef uBlasSparseMatrix   DefaultMatrix;
  typedef uBlasLUSolver       DefaultLUSolver;
  typedef uBlasKrylovSolver   DefaultKrylovSolver;
  typedef uBlasPreconditioner DefaultPreconditioner;
#endif

}

#endif
