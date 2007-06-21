// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2006, 2007.
//
// First added:  2006-05-15
// Last changed: 2007-06-21

#ifndef __VECTOR_H
#define __VECTOR_H

#include <dolfin/PETScVector.h>
#include <dolfin/uBlasVector.h>

namespace dolfin
{

  /// Vector is a synonym PETScVector if PETSc is enabled, otherwise
  /// it's uBlasSparseVector.

#ifdef HAVE_PETSC_H
  typedef PETScVector Vector;
#else
  typedef uBlasVector Vector;
#endif


}

#endif
