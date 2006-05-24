// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-15
// Last changed: 2006-05-24

#ifndef __MATRIX_H
#define __MATRIX_H

#ifdef HAVE_PETSC_H

#include <dolfin/SparseMatrix.h>

namespace dolfin
{

  /// Matrix is a synonym for the default matrix type which is SparseMatrix.

  typedef SparseMatrix Matrix;

}

#endif

#endif
