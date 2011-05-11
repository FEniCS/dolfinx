// Copyright (C) 2005-2006 Ola Skavhaug
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg, 2009.
//
// First added:  2007-12-06
// Last changed: 2009-05-18

#ifdef HAS_PETSC

#include "SparsityPattern.h"
#include "PETScLUSolver.h"
#include "PETScMatrix.h"
#include "PETScVector.h"
#include "PETScFactory.h"

using namespace dolfin;

// Singleton instance
PETScFactory PETScFactory::factory;

//-----------------------------------------------------------------------------
PETScMatrix* PETScFactory::create_matrix() const
{
  return new PETScMatrix();
}
//-----------------------------------------------------------------------------
PETScVector* PETScFactory:: create_vector() const
{
  return new PETScVector("global");
}
//-----------------------------------------------------------------------------
PETScVector* PETScFactory:: create_local_vector() const
{
  return new PETScVector("local");
}
//-----------------------------------------------------------------------------
SparsityPattern* PETScFactory::create_pattern() const
{
  return new SparsityPattern;
}
//-----------------------------------------------------------------------------
PETScLUSolver* PETScFactory::create_lu_solver() const
{
  return new PETScLUSolver();
}
//-----------------------------------------------------------------------------
PETScKrylovSolver* PETScFactory::create_krylov_solver(std::string method,
                                                      std::string pc) const
{
  return new PETScKrylovSolver(method, pc);
}
//-----------------------------------------------------------------------------

#endif
