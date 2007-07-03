// Copyright (C) 2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-07-03
// Last changed:

#ifndef __DEFAULT_LA_TYPES_H
#define __DEFAULT_LA_TYPES_H

#include <dolfin/PETScVector.h>
#include <dolfin/PETScMatrix.h>
#include <dolfin/PETScLUSolver.h>
#include <dolfin/PETScKrylovSolver.h>

#include <dolfin/uBlasVector.h>
#include <dolfin/uBlasSparseMatrix.h>
#include <dolfin/uBlasLUSolver.h>
#include <dolfin/uBlasKrylovSolver.h>

namespace dolfin
{

  /// Various default linear algebra quantities are defined here.
  
#ifdef HAVE_PETSC_H
  typedef PETScVector        DefaultVector;
  typedef PETScMatrix        DefaultMatrix;
  typedef PETScLUSolver      DefaultLUSolver;
  typedef PETScKrylovSolver  DefaultKrylovSolver;
#else
  typedef uBlasVector        DefaultVector;
  typedef uBlasSparseMatrix  DefaultMatrix;
  typedef uBlasLUSolver      DefaultLUSolver;
  typedef uBlasKrylovSolver  DefaultKrylovSolver;
#endif

}

#endif
