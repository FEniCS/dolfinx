// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2006, 2007.
//
// First added:  2006-05-15
// Last changed: 2007-06-21

#ifndef __MATRIX_H
#define __MATRIX_H

#include <dolfin/PETScMatrix.h>
#include <dolfin/uBlasSparseMatrix.h>

namespace dolfin
{

  /// Matrix is a synonym PETScMatrix if PETSc is enabled, otherwise
  /// it's uBlasSparseMatrix.

#ifdef HAVE_PETSC_H
  typedef PETScMatrix Matrix;
#else
  typedef uBlasSparseMatrix Matrix;
#endif
}


#endif
