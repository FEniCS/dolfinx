// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-05-31
// Last changed:

#ifndef __KRYLOV_SOLVER_H
#define __KRYLOV_SOLVER_H

#ifdef HAVE_PETSC_H
#include <dolfin/PETScKrylovSolver.h>
#endif

#include <dolfin/uBlasKrylovSolver.h>

namespace dolfin
{

#ifdef HAVE_PETSC_H
  typedef PETScKrylovSolver KrylovSolver;
#else
  typedef uBlasKrylovSolver KrylovSolver;
#endif

}


#endif
