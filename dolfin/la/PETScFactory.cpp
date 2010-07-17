// Copyright (C) 2005-2006 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
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
  return new SparsityPattern(SparsityPattern::unsorted);
}
//-----------------------------------------------------------------------------
GenericLinearSolver* PETScFactory::create_lu_solver() const
{
  return new PETScLUSolver();
}
//-----------------------------------------------------------------------------

#endif
