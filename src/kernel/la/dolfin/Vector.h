// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2006-05-15
// Last changed: 2006-05-29

#ifndef __VECTOR_H
#define __VECTOR_H

#ifdef HAVE_PETSC_H
#include <dolfin/SparseVector.h>
#endif

#include <dolfin/DenseVector.h>


namespace dolfin
{

  /// Vector is a synonym for the default vector type which is SparseVector.

#ifdef HAVE_PETSC_H
  typedef SparseVector Vector;
#else
  typedef DenseVector Vector;
#endif
}


#endif
