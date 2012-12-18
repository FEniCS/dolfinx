// Copyright (C) 2010 Garth N. Wells
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
std::size_t UniqueIdGenerator::id()
{
  // Get ID
  const std::size_t _id = unique_id_generator.next_id;

  // Increment ID
  ++unique_id_generator.next_id;

  return _id;
}
//-----------------------------------------------------------------------------
