// Copyright (C) 2010 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "UniqueIdGenerator.h"

using namespace dolfinx;
using namespace dolfinx::common;

// Initialise static data
dolfinx::common::UniqueIdGenerator
    dolfinx::common::UniqueIdGenerator::unique_id_generator;

//-----------------------------------------------------------------------------
UniqueIdGenerator::UniqueIdGenerator() : _next_id(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::size_t UniqueIdGenerator::id()
{
  // Get ID
  const std::size_t _id = unique_id_generator._next_id;

  // Increment ID
  ++unique_id_generator._next_id;

  return _id;
}
//-----------------------------------------------------------------------------
