// Copyright (C) 2005-2006 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-12-06
// Last changed: 2007-12-07

#ifdef HAVE_PETSC_H

#include <dolfin/SparsityPattern.h>
#include <dolfin/PETScMatrix.h>
#include <dolfin/PETScVector.h>
#include <dolfin/PETScFactory.h>

using namespace dolfin;

SparsityPattern* PETScFactory::createPattern() const 
{
  return new SparsityPattern(); 
}

PETScMatrix* PETScFactory::createMatrix() const 
{ 
  PETScMatrix* pm = new PETScMatrix();
  return pm;
}

PETScVector* PETScFactory:: createVector() const 
{ 
  return new PETScVector(); 
}

PETScFactory PETScFactory::petscfactory;

#endif
