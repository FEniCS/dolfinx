// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <cmath>
#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_math.h>
#include <dolfin/Matrix.h>
#include <dolfin/Vector.h>
#include <dolfin/GridHierarchy.h>
#include <dolfin/MultigridSolver.h>
#include <cmath>

using namespace dolfin;

//-----------------------------------------------------------------------------
MultigridSolver::MultigridSolver(GridHierarchy& grids)
{
  this->grids = grids;
}
//-----------------------------------------------------------------------------
void MultigridSolver::solve(Matrix& A, Vector& x, const Vector& b) 
{
}
//-----------------------------------------------------------------------------
