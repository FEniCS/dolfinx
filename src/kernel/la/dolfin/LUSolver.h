// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-31
// Last changed: 2006-08-15

#ifndef __LU_SOLVER_H
#define __LU_SOLVER_H

#ifdef HAVE_PETSC_H
#include <dolfin/PETScLUSolver.h>
#endif

#include <dolfin/uBlasLUSolver.h>

namespace dolfin
{

#ifdef HAVE_PETSC_H
  typedef PETScLUSolver LUSolver;
#else
  typedef uBlasLUSolver LUSolver;
#endif

}

#endif
