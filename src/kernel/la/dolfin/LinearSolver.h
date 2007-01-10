// Copyright (C) 2004-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2004-06-19
// Last changed: 2006-08-07

#ifndef __LINEAR_SOLVER_H
#define __LINEAR_SOLVER_H

#include <dolfin/PETScLinearSolver.h>
#include <dolfin/uBlasLinearSolver.h>

namespace dolfin
{

#ifdef HAVE_PETSC_H
  typedef PETScLinearSolver LinearSolver;
#else
  typedef uBlasLinearSolver LinearSolver;
#endif

}

#endif
