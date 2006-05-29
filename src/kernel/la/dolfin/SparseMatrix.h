// Copyright (C) 2006 Garth N. Wells
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-29
// Last changed:

#ifndef __SPARSE_MATRIX_H
#define __SPARSE_MATRIX_H

#ifdef HAVE_PETSC_H
#include <dolfin/PETScSparseMatrix.h>
#endif

#include <dolfin/uBlasSparseMatrix.h>

namespace dolfin
{

  /// SparseMatrix is a synonym PETScSparseMatrix if PETSc is enabled, otherwise 
  /// it's uBlasSparseMatrix.

#ifdef HAVE_PETSC_H
  typedef PETScSparseMatrix SparseMatrix;
#else
  typedef uBlasSparseMatrix SparseMatrix;
#endif

}


#endif
