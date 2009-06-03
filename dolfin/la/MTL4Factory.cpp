// Copyright (C) 2008 Dag Lindbo
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
// Modified by Garth N. Wells, 2009.
//
// First added:  2008-07-06
// Last changed: 2009-06-03

#ifdef HAS_MTL4

#include "MTL4Matrix.h"
#include "MTL4Vector.h"
#include "SparsityPattern.h"
#include "MTL4Factory.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MTL4Matrix* MTL4Factory::create_matrix() const
{
  return new MTL4Matrix();
}
//-----------------------------------------------------------------------------
MTL4Vector* MTL4Factory:: create_vector() const
{
  return new MTL4Vector();
}
//-----------------------------------------------------------------------------
SparsityPattern* MTL4Factory::create_pattern() const
{
  return new SparsityPattern(SparsityPattern::unsorted);
}
//-----------------------------------------------------------------------------

// Singleton instance
MTL4Factory MTL4Factory::factory;

#endif
