// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-15
// Last changed: 2006-05-24

#ifndef __VECTOR_H
#define __VECTOR_H

#ifdef HAVE_PETSC_H

#include <dolfin/SparseVector.h>

namespace dolfin
{

  /// Vector is a synonym for the default vector type which is SparseVector.

  typedef SparseVector Vector;

}

#endif

#endif
