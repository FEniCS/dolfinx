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

#ifndef __UNIQUE_ID_GENERATOR_H
#define __UNIQUE_ID_GENERATOR_H

#include <cstddef>

namespace dolfin
{

  // FIXME: Make a base class that classes can inherit from

  /// This is a singleton class that return IDs that are unique in the
  /// lifetime of a program.

  class UniqueIdGenerator
  {
  public:

    UniqueIdGenerator();

    /// Generate a unique ID
    static std::size_t id();

  private:

    // Singleton instance
    static UniqueIdGenerator unique_id_generator;

    // Next ID to be returned
    std::size_t next_id;

  };

}

#endif
