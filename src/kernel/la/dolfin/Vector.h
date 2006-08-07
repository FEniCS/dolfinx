// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2006-05-15
// Last changed: 2006-05-31

#ifndef __VECTOR_H
#define __VECTOR_H

#ifdef HAVE_PETSC_H
#include <dolfin/PETScVector.h>
#endif

#include <dolfin/uBlasVector.h>


namespace dolfin
{

  /// Vector is a synonym for the default vector type which is PETScVector if
  /// PETSc is enabled, otherwise it is uBlasVector.

#ifdef HAVE_PETSC_H
  typedef PETScVector Vector;
#else
  typedef uBlasVector Vector;
#endif
}


#endif
