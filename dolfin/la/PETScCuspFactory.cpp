// Copyright (C) 2011 Fredrik Valdmanis
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2011-09-13 
// Last changed: 2011-09-13

#ifdef PETSC_HAVE_CUSP

#include "SparsityPattern.h"
#include "PETScLUSolver.h"
#include "PETScCuspMatrix.h"
#include "PETScCuspVector.h"
#include "PETScCuspFactory.h"

using namespace dolfin;

// Singleton instance
PETScCuspFactory PETScCuspFactory::factory;

//-----------------------------------------------------------------------------
PETScCuspMatrix* PETScCuspFactory::create_matrix() const
{
  return new PETScCuspMatrix();
}
//-----------------------------------------------------------------------------
PETScCuspVector* PETScCuspFactory:: create_vector() const
{
  return new PETScCuspVector("global");
}
//-----------------------------------------------------------------------------
PETScCuspVector* PETScCuspFactory:: create_local_vector() const
{
  return new PETScCuspVector("local");
}
//-----------------------------------------------------------------------------
SparsityPattern* PETScCuspFactory::create_pattern() const
{
  return new SparsityPattern;
}
//-----------------------------------------------------------------------------
PETScLUSolver* PETScCuspFactory::create_lu_solver() const
{
  return new PETScLUSolver();
}
//-----------------------------------------------------------------------------
PETScKrylovSolver* PETScCuspFactory::create_krylov_solver(std::string method,
                                                      std::string pc) const
{
  return new PETScKrylovSolver(method, pc);
}
//-----------------------------------------------------------------------------

#endif
