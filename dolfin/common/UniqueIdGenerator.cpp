// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-12-05
// Last changed:

#include "UniqueIdGenerator.h"

using namespace dolfin;

// Initialise static data
dolfin::UniqueIdGenerator dolfin::UniqueIdGenerator::unique_id_generator;

//-----------------------------------------------------------------------------
UniqueIdGenerator::UniqueIdGenerator() : next_id(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::uint UniqueIdGenerator::id()
{
  // Get ID
  const uint _id = unique_id_generator.next_id;

  // Increment ID
  ++unique_id_generator.next_id;

  return _id;
}
//-----------------------------------------------------------------------------
