// Copyright (C) 2008 Dag Lindbo
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-07-06

#ifdef HAS_MTL4

#include "MTL4Matrix.h"
#include "MTL4Vector.h"
#include "MTL4Factory.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MTL4Factory:: MTL4Factory()
{

}
//-----------------------------------------------------------------------------
MTL4Matrix* MTL4Factory::createMatrix() const 
{ 
  return new MTL4Matrix();
}
//-----------------------------------------------------------------------------
MTL4SparsityPattern* MTL4Factory::createPattern() const 
{
  return new MTL4SparsityPattern();
}
//-----------------------------------------------------------------------------
MTL4Vector* MTL4Factory::createVector() const 
{ 
  return new MTL4Vector(); 
}

// Singleton instance
MTL4Factory MTL4Factory::factory;


#endif
