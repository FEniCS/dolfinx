// Copyright (C) 2007 Ilmar Wilbers.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-05-21
// Last changed: 2008-05-22

#include "SparsityPattern.h"
#include "AssemblyMatrix.h"
#include "uBlasVector.h"
#include "AssemblyFactory.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
SparsityPattern* AssemblyFactory::createPattern() const 
{
  return new SparsityPattern(); 
}
//-----------------------------------------------------------------------------
AssemblyMatrix* AssemblyFactory::createMatrix() const
{ 
  return new AssemblyMatrix(); 
}
//-----------------------------------------------------------------------------
uBlasVector* AssemblyFactory::createVector() const
{ 
  return new uBlasVector(); 
}
//-----------------------------------------------------------------------------

// Singleton instance
AssemblyFactory AssemblyFactory::factory;
