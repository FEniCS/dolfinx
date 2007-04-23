// Copyright (C) 2006 Garth N. Wells
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-05-29
// Last changed:

#ifndef __SPARSE_MATRIX_H
#define __SPARSE_MATRIX_H

#ifdef HAVE_PETSC_H
#include <dolfin/PETScMatrix.h>
#endif

#include <dolfin/uBlasSparseMatrix.h>

namespace dolfin
{

  /// SparseMatrix is a synonym PETScMatrix if PETSc is enabled, otherwise 
  /// it's uBlasSparseMatrix.

#ifdef HAVE_PETSC_H
  typedef PETScMatrix SparseMatrix;
#else
  typedef uBlasSparseMatrix SparseMatrix;
#endif

}


#endif
