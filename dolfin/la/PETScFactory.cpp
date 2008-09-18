// Copyright (C) 2005-2006 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-12-06
// Last changed: 2007-12-07

#ifdef HAS_PETSC

#include "SparsityPattern.h"
#include "PETScMatrix.h"
#include "PETScVector.h"
#include "PETScFactory.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
SparsityPattern* PETScFactory::create_pattern() const 
{
  return new SparsityPattern(); 
}
//-----------------------------------------------------------------------------
PETScMatrix* PETScFactory::create_matrix() const 
{ 
  PETScMatrix* pm = new PETScMatrix();
  return pm;
}
//-----------------------------------------------------------------------------
PETScVector* PETScFactory:: create_vector() const 
{ 
  return new PETScVector(); 
}
//-----------------------------------------------------------------------------

// Singleton instance
PETScFactory PETScFactory::factory;

#endif
